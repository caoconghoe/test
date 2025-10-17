# desktop_app.py
"""
dịt cụ
Face Attendance / Timekeeping - Desktop App (Tkinter)
-----------------------------------------------------
This converts the Streamlit web app logic into a native desktop GUI.

Features
- Face recognition check-in/out via webcam
- Employee CRUD (add from image file)
- Working-hours report (per employee per day; hours = last - first scan)
- CSV export

Dependencies
- python -m pip install opencv-python face_recognition numpy pandas pillow
- (face_recognition requires dlib; on Windows consider conda install -c conda-forge dlib)

Run
- python desktop_app.py
- Optional: build an .exe with PyInstaller:
  pyinstaller --onefile --noconsole --name FaceAttendance desktop_app.py
"""
import io
import os
import sqlite3
import datetime as dt
from typing import Optional, Tuple, List
import threading

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# Optional OpenCV import (for webcam)
try:
    import cv2
except Exception:
    cv2 = None

try:
    import face_recognition
    _face_err = None
except Exception as e:
    face_recognition = None
    _face_err = e

# ---------------------------
# Config
# ---------------------------
APP_TITLE = "Face Attendance - Desktop"
DATA_DIR = os.environ.get("FA_DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "face_attendance.db")
DEFAULT_TOL = 0.45  # lower = stricter

# ---------------------------
# Database helpers
# ---------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_code TEXT UNIQUE,
            name TEXT NOT NULL,
            department TEXT,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_id INTEGER NOT NULL,
            ts TEXT NOT NULL,
            device TEXT,
            FOREIGN KEY(emp_id) REFERENCES employees(id)
        );
        """
    )
    con.commit()
    con.close()

def np_to_blob(vec: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, vec.astype(np.float32))
    return buf.getvalue()

def blob_to_np(blob: bytes) -> np.ndarray:
    buf = io.BytesIO(blob)
    buf.seek(0)
    return np.load(buf, allow_pickle=False)

def load_all_embeddings() -> pd.DataFrame:
    con = get_conn()
    df = pd.read_sql_query(
        "SELECT id, emp_code, name, department, embedding FROM employees", con
    )
    con.close()
    if not df.empty:
        df["embedding_vec"] = df["embedding"].apply(blob_to_np)
    else:
        df["embedding_vec"] = []
    return df

def face_encode_from_image(img: Image.Image) -> Optional[np.ndarray]:
    if face_recognition is None:
        raise RuntimeError(f"face_recognition not available: {_face_err}")
    rgb = np.array(img.convert("RGB"))
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes:
        return None
    encs = face_recognition.face_encodings(rgb, known_face_locations=boxes)
    if len(encs) == 0:
        return None
    return encs[0]

def match_employee(embedding: np.ndarray, tol: float = DEFAULT_TOL) -> Optional[dict]:
    if face_recognition is None:
        raise RuntimeError(f"face_recognition not available: {_face_err}")
    df = load_all_embeddings()
    if df.empty:
        return None
    known = np.stack(df["embedding_vec"].to_list(), axis=0)
    dists = face_recognition.face_distance(known, embedding)
    idx = int(np.argmin(dists))
    if dists[idx] < tol:
        row = df.iloc[idx].to_dict()
        row["distance"] = float(dists[idx])
        return row
    return None

def add_employee(emp_code: str, name: str, department: str, embedding: np.ndarray) -> Tuple[bool, str]:
    con = get_conn()
    cur = con.cursor()
    try:
        cur.execute(
            "INSERT INTO employees(emp_code, name, department, embedding, created_at) VALUES(?,?,?,?,?)",
            (emp_code, name, department, np_to_blob(embedding), dt.datetime.now().isoformat()),
        )
        con.commit()
        return True, "Đã thêm nhân viên."
    except sqlite3.IntegrityError:
        return False, "Mã nhân viên đã tồn tại."
    except Exception as e:
        return False, f"Lỗi: {e}"
    finally:
        con.close()

def delete_employee(emp_id: int) -> None:
    con = get_conn()
    cur = con.cursor()
    cur.execute("DELETE FROM employees WHERE id=?", (emp_id,))
    con.commit()
    con.close()

def mark_attendance(emp_id: int, device: str = "desktop") -> None:
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO attendance(emp_id, ts, device) VALUES(?,?,?)",
        (emp_id, dt.datetime.now().isoformat(), device),
    )
    con.commit()
    con.close()

def get_attendance(start: Optional[dt.date] = None, end: Optional[dt.date] = None) -> pd.DataFrame:
    con = get_conn()
    q = "SELECT emp_id, ts, device FROM attendance"
    params: List[str] = []
    if start and end:
        q += " WHERE date(ts) BETWEEN ? AND ?"
        params.extend([start.isoformat(), end.isoformat()])
    elif start:
        q += " WHERE date(ts) >= ?"
        params.append(start.isoformat())
    elif end:
        q += " WHERE date(ts) <= ?"
        params.append(end.isoformat())
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    if df.empty:
        return df
    # Join employee info
    df_emp = load_all_embeddings()
    id2name = df_emp.set_index("id")["name"].to_dict()
    id2dept = df_emp.set_index("id")["department"].to_dict()
    df["name"] = df["emp_id"].map(id2name)
    df["department"] = df["emp_id"].map(id2dept)
    df["ts"] = pd.to_datetime(df["ts"])
    return df

def compute_work_hours(att_df: pd.DataFrame) -> pd.DataFrame:
    if att_df.empty:
        return att_df
    df = att_df.copy()
    df["date"] = df["ts"].dt.date
    agg = (
        df.sort_values(["emp_id", "ts"])
        .groupby(["emp_id", "name", "department", "date"])
        .agg(first_in=("ts", "min"), last_out=("ts", "max"), scans=("ts", "count"))
        .reset_index()
    )
    agg["hours"] = (agg["last_out"] - agg["first_in"]).dt.total_seconds() / 3600.0
    agg.loc[agg["scans"] <= 1, "hours"] = 0.0
    return agg

# ---------------------------
# Tkinter GUI
# ---------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x680")
        self.resizable(True, True)

        if face_recognition is None:
            messagebox.showerror("Lỗi", f"Không tải được face_recognition: {_face_err}")
            self.destroy()
            return

        init_db()

        self.notebook = ttk.Notebook(self)
        self.tab_att = ttk.Frame(self.notebook)
        self.tab_emp = ttk.Frame(self.notebook)
        self.tab_rep = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_att, text="Chấm công")
        self.notebook.add(self.tab_emp, text="Nhân viên")
        self.notebook.add(self.tab_rep, text="Báo cáo")
        self.notebook.pack(fill="both", expand=True)

        self.build_attendance_tab()
        self.build_employee_tab()
        self.build_report_tab()

        # Webcam state
        self.cap = None
        self._video_loop = False
        self._frame_imgtk = None

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # --------------- Attendance Tab ---------------
    def build_attendance_tab(self):
        frm = self.tab_att

        top = ttk.Frame(frm)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="Ngưỡng nhận diện (nhỏ = chặt):").pack(side="left")
        self.scale_tol = ttk.Scale(top, from_=0.30, to=0.70, value=DEFAULT_TOL, orient="horizontal", length=200)
        self.scale_tol.pack(side="left", padx=6)

        ttk.Button(top, text="Bật camera", command=self.start_camera).pack(side="left", padx=4)
        ttk.Button(top, text="Tắt camera", command=self.stop_camera).pack(side="left", padx=4)
        ttk.Button(top, text="Chấm công (chụp)", command=self.scan_and_mark).pack(side="left", padx=4)

        self.video_panel = ttk.Label(frm, relief="sunken", anchor="center")
        self.video_panel.pack(fill="both", expand=True, padx=10, pady=10)

        self.att_status = tk.StringVar(value="Chưa có thao tác.")
        ttk.Label(frm, textvariable=self.att_status).pack(pady=6)

    def start_camera(self):
        if cv2 is None:
            messagebox.showerror("Lỗi", "OpenCV (cv2) chưa được cài đặt.")
            return
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror("Lỗi", "Không mở được webcam.")
            return
        self._video_loop = True
        threading.Thread(target=self._update_video, daemon=True).start()
        self.att_status.set("Camera đã bật.")

    def stop_camera(self):
        self._video_loop = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.video_panel.config(image="", text="")
        self.att_status.set("Camera đã tắt.")

    def _update_video(self):
        while self._video_loop and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Convert BGR -> RGB for display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((900, 520))
            imgtk = ImageTk.PhotoImage(image=img)
            # Keep reference
            self._frame_imgtk = imgtk
            self.video_panel.config(image=imgtk)
        # loop exits

    def scan_and_mark(self):
        tol = float(self.scale_tol.get())
        if self.cap is None:
            messagebox.showwarning("Chú ý", "Hãy bật camera trước.")
            return
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không đọc được frame từ camera.")
            return
        # Encode
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        try:
            enc = face_encode_from_image(img)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi xử lý ảnh: {e}")
            return
        if enc is None:
            self.att_status.set("Không phát hiện được khuôn mặt. Thử lại.")
            return
        m = match_employee(enc, tol)
        if m is None:
            self.att_status.set("❌ Không khớp với nhân viên nào (Unknown). Vào tab Nhân viên để thêm.")
        else:
            mark_attendance(int(m["id"]))
            self.att_status.set(f"✅ Đã chấm công cho {m['name']} (khoảng cách={m['distance']:.3f}).")

    # --------------- Employee Tab ---------------
    def build_employee_tab(self):
        frm = self.tab_emp

        # List
        list_frame = ttk.Frame(frm)
        list_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        ttk.Label(list_frame, text="Danh sách nhân viên").pack(anchor="w")
        self.tree = ttk.Treeview(list_frame, columns=("id","code","name","dept"), show="headings", height=22)
        self.tree.heading("id", text="ID")
        self.tree.heading("code", text="Mã NV")
        self.tree.heading("name", text="Họ tên")
        self.tree.heading("dept", text="Phòng ban")
        self.tree.column("id", width=60, anchor="center")
        self.tree.column("code", width=100)
        self.tree.column("name", width=180)
        self.tree.column("dept", width=120)
        self.tree.pack(fill="both", expand=True)

        btns = ttk.Frame(list_frame)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Tải lại", command=self.refresh_employees).pack(side="left", padx=4)
        ttk.Button(btns, text="Xoá nhân viên đã chọn", command=self.delete_selected_employee).pack(side="left", padx=4)

        # Add form
        form = ttk.LabelFrame(frm, text="Thêm nhân viên mới")
        form.pack(side="right", fill="y", padx=10, pady=10)

        self.emp_code_var = tk.StringVar()
        self.emp_name_var = tk.StringVar()
        self.emp_dept_var = tk.StringVar()
        ttk.Label(form, text="Mã nhân viên").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(form, textvariable=self.emp_code_var, width=26).grid(row=0, column=1, padx=6, pady=4)
        ttk.Label(form, text="Họ tên").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(form, textvariable=self.emp_name_var, width=26).grid(row=1, column=1, padx=6, pady=4)
        ttk.Label(form, text="Phòng ban").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(form, textvariable=self.emp_dept_var, width=26).grid(row=2, column=1, padx=6, pady=4)

        self.face_path_var = tk.StringVar()
        ttk.Label(form, text="Ảnh khuôn mặt (jpg/png)").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        entry = ttk.Entry(form, textvariable=self.face_path_var, width=26)
        entry.grid(row=3, column=1, padx=6, pady=4)
        ttk.Button(form, text="Chọn ảnh...", command=self.browse_face_image).grid(row=3, column=2, padx=6, pady=4)

        ttk.Button(form, text="Lưu nhân viên", command=self.save_employee_from_file).grid(row=4, column=1, pady=10)

        for i in range(4):
            form.grid_rowconfigure(i, weight=0)
        form.grid_columnconfigure(1, weight=1)

        self.refresh_employees()

    def refresh_employees(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        df = load_all_embeddings()
        if df.empty:
            return
        for _, r in df.iterrows():
            self.tree.insert("", "end", values=(int(r["id"]), r["emp_code"], r["name"], r["department"]))

    def browse_face_image(self):
        fpath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if fpath:
            self.face_path_var.set(fpath)

    def save_employee_from_file(self):
        code = self.emp_code_var.get().strip()
        name = self.emp_name_var.get().strip()
        dept = self.emp_dept_var.get().strip()
        fpath = self.face_path_var.get().strip()
        if not code or not name or not fpath:
            messagebox.showwarning("Thiếu thông tin", "Nhập đủ Mã NV, Họ tên và chọn ảnh.")
            return
        try:
            img = Image.open(fpath).convert("RGB")
            enc = face_encode_from_image(img)
            if enc is None:
                messagebox.showerror("Lỗi", "Không phát hiện được khuôn mặt trong ảnh.")
                return
            ok, msg = add_employee(code, name, dept, enc)
            if ok:
                messagebox.showinfo("Thành công", msg)
                self.refresh_employees()
                self.emp_code_var.set("")
                self.emp_name_var.set("")
                self.emp_dept_var.set("")
                self.face_path_var.set("")
            else:
                messagebox.showerror("Lỗi", msg)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi xử lý: {e}")

    def delete_selected_employee(self):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        emp_id = int(vals[0])
        if messagebox.askyesno("Xác nhận", f"Xoá nhân viên ID {emp_id}?"):
            delete_employee(emp_id)
            self.refresh_employees()

    # --------------- Reports Tab ---------------
    def build_report_tab(self):
        frm = self.tab_rep

        top = ttk.Frame(frm)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="Từ ngày (YYYY-MM-DD):").pack(side="left")
        self.start_var = tk.StringVar(value=str(dt.date.today().replace(day=1)))
        ttk.Entry(top, textvariable=self.start_var, width=12).pack(side="left", padx=6)

        ttk.Label(top, text="Đến ngày (YYYY-MM-DD):").pack(side="left")
        self.end_var = tk.StringVar(value=str(dt.date.today()))
        ttk.Entry(top, textvariable=self.end_var, width=12).pack(side="left", padx=6)

        ttk.Button(top, text="Lấy dữ liệu", command=self.load_reports).pack(side="left", padx=6)
        ttk.Button(top, text="Xuất CSV (công theo ngày)", command=self.export_daily_csv).pack(side="left", padx=6)
        ttk.Button(top, text="Xuất CSV (tổng theo NV)", command=self.export_total_csv).pack(side="left", padx=6)

        # Treeviews
        ttk.Label(frm, text="Công theo ngày (mỗi người mỗi ngày)").pack(anchor="w", padx=10)
        cols_daily = ("emp_id","name","department","date","first_in","last_out","scans","hours")
        self.tree_daily = ttk.Treeview(frm, columns=cols_daily, show="headings", height=10)
        for c in cols_daily:
            self.tree_daily.heading(c, text=c)
            self.tree_daily.column(c, width=120, anchor="center")
        self.tree_daily.pack(fill="x", padx=10, pady=6)

        ttk.Label(frm, text="Tổng hợp theo nhân viên").pack(anchor="w", padx=10)
        cols_total = ("emp_id","name","department","days","total_hours","scans")
        self.tree_total = ttk.Treeview(frm, columns=cols_total, show="headings", height=10)
        for c in cols_total:
            self.tree_total.heading(c, text=c)
            self.tree_total.column(c, width=120, anchor="center")
        self.tree_total.pack(fill="x", padx=10, pady=6)

        self._daily_df = None
        self._total_df = None

    def parse_date(self, s: str) -> Optional[dt.date]:
        try:
            return dt.datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None

    def load_reports(self):
        s = self.parse_date(self.start_var.get().strip())
        e = self.parse_date(self.end_var.get().strip())
        if s is None or e is None:
            messagebox.showwarning("Sai định dạng", "Ngày phải theo định dạng YYYY-MM-DD.")
            return
        att = get_attendance(s, e)
        if att.empty:
            messagebox.showinfo("Thông báo", "Không có dữ liệu trong khoảng ngày đã chọn.")
            self._daily_df = None
            self._total_df = None
            self._populate_tree(self.tree_daily, [])
            self._populate_tree(self.tree_total, [])
            return
        daily = compute_work_hours(att)

        # Fill daily tree
        rows_daily = []
        for _, r in daily.iterrows():
            rows_daily.append([
                int(r["emp_id"]), r["name"], r["department"],
                str(r["date"]), str(r["first_in"]), str(r["last_out"]),
                int(r["scans"]), round(float(r["hours"]), 2)
            ])
        self._populate_tree(self.tree_daily, rows_daily)

        total = (
            daily.groupby(["emp_id", "name", "department"]).agg(
                days=("date", "nunique"),
                total_hours=("hours", "sum"),
                scans=("scans", "sum"),
            ).reset_index()
        )
        total["total_hours"] = total["total_hours"].round(2)

        # Fill total tree
        rows_total = []
        for _, r in total.iterrows():
            rows_total.append([
                int(r["emp_id"]), r["name"], r["department"],
                int(r["days"]), float(r["total_hours"]), int(r["scans"])
            ])
        self._populate_tree(self.tree_total, rows_total)

        self._daily_df = daily
        self._total_df = total

    def _populate_tree(self, tree: ttk.Treeview, rows):
        for i in tree.get_children():
            tree.delete(i)
        for row in rows:
            tree.insert("", "end", values=row)

    def export_daily_csv(self):
        if self._daily_df is None or self._daily_df.empty:
            messagebox.showwarning("Chú ý", "Không có dữ liệu để xuất.")
            return
        fpath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not fpath:
            return
        df = self._daily_df.copy()
        df.to_csv(fpath, index=False, encoding="utf-8-sig")
        messagebox.showinfo("OK", f"Đã lưu: {fpath}")

    def export_total_csv(self):
        if self._total_df is None or self._total_df.empty:
            messagebox.showwarning("Chú ý", "Không có dữ liệu để xuất.")
            return
        fpath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not fpath:
            return
        df = self._total_df.copy()
        df.to_csv(fpath, index=False, encoding="utf-8-sig")
        messagebox.showinfo("OK", f"Đã lưu: {fpath}")

    def on_close(self):
        self._video_loop = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.destroy()

def main():
    app = App()
    if app.winfo_exists():
        app.mainloop()

if __name__ == "__main__":
    main()
