import tkinter as tk
from tkinter import filedialog, messagebox
import traceback
import neuralaudiostyle

def start_style_transfer():
    try:
        content = content_entry.get()
        style = style_entry.get()
        output = output_entry.get()
        neuralaudiostyle.perform_style_transfer(content, style, output)
        messagebox.showinfo("Success", "Style transfer complete!")
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        error_traceback = traceback.format_exc()
        print(error_message)
        print(error_traceback)
        messagebox.showerror("Error", error_message)

def select_content_file():
    file_path = filedialog.askopenfilename()
    content_entry.delete(0, tk.END)
    content_entry.insert(0, file_path)

def select_style_file():
    file_path = filedialog.askopenfilename()
    style_entry.delete(0, tk.END)
    style_entry.insert(0, file_path)

root = tk.Tk()
root.title("Neural Song Style Transfer")

tk.Label(root, text="Content Audio File:").grid(row=0, column=0, padx=10, pady=10)
content_entry = tk.Entry(root, width=50)
content_entry.grid(row=0, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=select_content_file).grid(row=0, column=2, padx=10, pady=10)

tk.Label(root, text="Style Audio File:").grid(row=1, column=0, padx=10, pady=10)
style_entry = tk.Entry(root, width=50)
style_entry.grid(row=1, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=select_style_file).grid(row=1, column=2, padx=10, pady=10)

tk.Label(root, text="Output File Path:").grid(row=2, column=0, padx=10, pady=10)
output_entry = tk.Entry(root, width=50)
output_entry.grid(row=2, column=1, padx=10, pady=10)

tk.Button(root, text="Start Style Transfer", command=start_style_transfer).grid(row=3, column=0, columnspan=3, padx=10, pady=10)

root.mainloop()
