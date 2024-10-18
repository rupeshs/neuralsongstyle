import tkinter as tk
from tkinter import filedialog
import neuralaudiostyle

class NeuralStyleTransferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Song Style Transfer")

        self.content_path = tk.StringVar()
        self.style_path = tk.StringVar()
        self.output_path = tk.StringVar()

        tk.Label(root, text="Content Audio File:").grid(row=0, column=0, padx=10, pady=10)
        tk.Entry(root, textvariable=self.content_path, width=50).grid(row=0, column=1, padx=10, pady=10)
        tk.Button(root, text="Browse", command=self.browse_content).grid(row=0, column=2, padx=10, pady=10)

        tk.Label(root, text="Style Audio File:").grid(row=1, column=0, padx=10, pady=10)
        tk.Entry(root, textvariable=self.style_path, width=50).grid(row=1, column=1, padx=10, pady=10)
        tk.Button(root, text="Browse", command=self.browse_style).grid(row=1, column=2, padx=10, pady=10)

        tk.Label(root, text="Output File Path:").grid(row=2, column=0, padx=10, pady=10)
        tk.Entry(root, textvariable=self.output_path, width=50).grid(row=2, column=1, padx=10, pady=10)
        tk.Button(root, text="Browse", command=self.browse_output).grid(row=2, column=2, padx=10, pady=10)

        self.progress_label = tk.Label(root, text="")
        self.progress_label.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

        tk.Button(root, text="Start Style Transfer", command=self.start_style_transfer).grid(row=4, column=0, columnspan=3, padx=10, pady=10)

    def browse_content(self):
        self.content_path.set(filedialog.askopenfilename())

    def browse_style(self):
        self.style_path.set(filedialog.askopenfilename())

    def browse_output(self):
        self.output_path.set(filedialog.asksaveasfilename(defaultextension=".wav"))

    def start_style_transfer(self):
        content = self.content_path.get()
        style = self.style_path.get()
        output = self.output_path.get()

        if content and style and output:
            self.progress_label.config(text="Style transfer in progress...")
            self.root.update_idletasks()
            neuralaudiostyle.perform_style_transfer(content, style, output)
            self.progress_label.config(text="Style transfer completed!")
        else:
            self.progress_label.config(text="Please select all files.")

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralStyleTransferApp(root)
    root.mainloop()
