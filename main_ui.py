import customtkinter as ctk
import subprocess
import threading
import sys
import time

# ---------- APP CONFIG ----------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.geometry("800x500")
root.title("Sign Language Recognition System")

# ---------- TITLE ----------
title = ctk.CTkLabel(root, 
                     text="🤟 Sign Language to Text & Speech", 
                     font=("Poppins", 28, "bold"))
title.pack(pady=(40, 20))

# ---------- DESCRIPTION ----------
desc = ctk.CTkLabel(root, 
                    text="Convert real-time sign language gestures to text and speech using AI.",
                    font=("Poppins", 14), 
                    text_color="#A9A9A9")
desc.pack(pady=(0, 20))

# ---------- BUTTON FUNCTION ----------
def run_app():
    status_label.configure(text="🔄 Initializing webcam and model...", text_color="#00BFFF")

    def launch():
        python_executable = sys.executable
        try:
            process = subprocess.Popen(
                [python_executable, "final_pred.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Wait until final_pred prints "WEB_CAM_READY"
            for line in iter(process.stdout.readline, ''):
                if "WEB_CAM_READY" in line:
                    root.after(0, root.destroy)
                    return

            # If process ends without printing the signal → show error
            stdout, stderr = process.communicate(timeout=2)
            error_message = stderr.strip() or "Unknown startup issue."
            root.after(0, lambda: status_label.configure(
                text=f"❌ Failed to start webcam:\n{error_message}",
                text_color="red"
            ))

        except Exception as e:
            root.after(0, lambda: status_label.configure(
                text=f"❌ Unexpected Error: {e}",
                text_color="red"
            ))

    threading.Thread(target=launch, daemon=True).start()

# ---------- MAIN BUTTON ----------
btn_start = ctk.CTkButton(root, 
                          text="Start Sign Recognition", 
                          fg_color="#0078FF", 
                          hover_color="#005FCC", 
                          font=("Poppins", 18, "bold"), 
                          width=240, height=50, 
                          corner_radius=15,
                          command=run_app)
btn_start.pack(pady=(30, 15))

# ---------- STATUS ----------
status_label = ctk.CTkLabel(root, text="", font=("Poppins", 14))
status_label.pack(pady=(5, 20))

# ---------- FOOTER ----------
footer = ctk.CTkLabel(root, 
                      text="Developed by Lavanya Pitta", 
                      font=("Poppins", 12), 
                      text_color="#888")
footer.pack(side="bottom", pady=10)

root.mainloop()
