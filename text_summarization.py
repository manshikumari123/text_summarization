import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class TextSummarizationApp:
    def __init__(self, master):
        self.master = master
        master.title("Text Summarization Tool")
        
        # Load models
        self.models = {
            "BART": "facebook/bart-large-cnn",
            "T5": "t5-base"
        }
        self.model_name = "facebook/bart-large-cnn"  # Default model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Create GUI components
        self.label = tk.Label(master, text="Enter text to summarize:")
        self.label.pack()

        self.text_area = scrolledtext.ScrolledText(master, width=50, height=10)
        self.text_area.pack()

        self.summarize_button = tk.Button(master, text="Summarize", command=self.summarize_text)
        self.summarize_button.pack()

        self.result_label = tk.Label(master, text="Summary:")
        self.result_label.pack()

        self.summary_area = scrolledtext.ScrolledText(master, width=50, height=10)
        self.summary_area.pack()

        self.load_model_button = tk.Button(master, text="Load Model", command=self.load_model)
        self.load_model_button.pack()

        self.save_button = tk.Button(master, text="Save Summary", command=self.save_summary)
        self.save_button.pack()

    def load_model(self):
        # Load the selected model based on user input
        model_choice = tk.simpledialog.askstring("Model Selection", "Enter model name (BART/T5):")
        if model_choice in self.models:
            self.model_name = self.models[model_choice]
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            messagebox.showinfo("Success", f"Loaded model: {self.model_name}")
        else:
            messagebox.showerror("Error", "Invalid model name. Please choose either BART or T5.")

    def summarize_text(self):
        text = self.text_area.get("1.0", tk.END).strip()
        if len(text) < 50:
            messagebox.showerror("Error", "Input text is too short for summarization. Please provide a longer text.")
            return

        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs['input_ids'], max_length=130, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        self.summary_area.delete("1.0", tk.END)  # Clear previous summary
        self.summary_area.insert(tk.END, summary)  # Insert new summary

    def save_summary(self):
        summary = self.summary_area.get("1.0", tk.END).strip()
        if not summary:
            messagebox.showerror("Error", "No summary to save.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            messagebox.showinfo("Success", f"Summary saved to {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TextSummarizationApp(root)
    root.mainloop()
