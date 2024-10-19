import tkinter as tk
from tkinter import filedialog, messagebox
import neuralaudiostyle

def select_file(entry):
    file_path = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

def start_style_transfer(content_entry, style_entry, output_entry):
    content = content_entry.get()
    style = style_entry.get()
    output = output_entry.get()

    try:
        neuralaudiostyle.perform_style_transfer(content, style, output)
        messagebox.showinfo("Success", "Style transfer complete!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def create_gui():
    root = tk.Tk()
    root.title("Neural Song Style Transfer")

    tk.Label(root, text="Content Audio File:").grid(row=0, column=0, padx=10, pady=10)
    content_entry = tk.Entry(root, width=50)
    content_entry.grid(row=0, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse", command=lambda: select_file(content_entry)).grid(row=0, column=2, padx=10, pady=10)

    tk.Label(root, text="Style Audio File:").grid(row=1, column=0, padx=10, pady=10)
    style_entry = tk.Entry(root, width=50)
    style_entry.grid(row=1, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse", command=lambda: select_file(style_entry)).grid(row=1, column=2, padx=10, pady=10)

    tk.Label(root, text="Output File Path:").grid(row=2, column=0, padx=10, pady=10)
    output_entry = tk.Entry(root, width=50)
    output_entry.grid(row=2, column=1, padx=10, pady=10)

    tk.Button(root, text="Start Style Transfer", command=lambda: start_style_transfer(content_entry, style_entry, output_entry)).grid(row=3, column=0, columnspan=3, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
