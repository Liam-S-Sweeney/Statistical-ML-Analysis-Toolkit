import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from config import DATA_PATH, SINGLE_VAR_DESCRIPTIVES_PATH, MULTIVARIATE_ANALYSIS_PATH
from data_loader import load_clean
import os
from global_descriptive_generator import master_descriptive_csv_generator, all_single_var_descriptive_csv_generator
from multivariate_exploration import explore_multi_variables, multivariate_visualizations, correlational_analysis
from gmm_analysis import gmm_analysis

# Dataframe Selection

if not Path(DATA_PATH).exists() and not Path(SINGLE_VAR_DESCRIPTIVES_PATH).exists() and not Path(MULTIVARIATE_ANALYSIS_PATH).exists():
    os.makedirs("data_files", exist_ok=True)
    os.makedirs('single_var_descriptives', exist_ok=True)
    os.makedirs("multivariate_analysis", exist_ok=True)
    print(f"CSV file not found at {DATA_PATH}\n"
    "Please place the dataset inside the 'data_files' folder.")
    exit()

df = load_clean()

# All variable options
var_options = [var for var in df.columns]


# Create window
root = tk.Tk()
root.title("Statistical + ML Analysis Tool")
root.geometry('500x500')


# Root layout
root.columnconfigure(0, weight=1)
root.rowconfigure(1, weight=1)

top_frame = ttk.Frame(root)
top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
top_frame.columnconfigure(0, weight=1)
top_frame.columnconfigure(1, weight=1)

dropdown_frame = ttk.Frame(root)
dropdown_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
dropdown_frame.columnconfigure(0, weight=1)

bottom_frame = ttk.Frame(root)
bottom_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
bottom_frame.columnconfigure(0, weight=1)
bottom_frame.columnconfigure(1, weight=1)


# Create variable combobox class
class SearchableDropdown:
    def __init__(self, parent, options, row):
        self.str_var = tk.StringVar()
        self.options = options
        self.dropdown = ttk.Combobox(parent, values=options, textvariable=self.str_var)
        self.dropdown.grid(row=row, column=0, pady=4, sticky="ew")
        self.str_var.trace_add("write", self.check_input)

    def check_input(self, *args):
        typed = self.str_var.get()

        if typed == "":
            self.dropdown['values'] = self.options
        else:
            filtered = [item for item in self.options if item.lower().startswith(typed.lower())]
            self.dropdown['values'] = filtered

    def get_value(self):
        return self.str_var.get().strip()


# Determine how many variables are wanted with spinbox
dropdown_widgets = []

def update_dropdowns(*args):
    global dropdown_widgets

    desired = int(vc_str_var.get())
    current = len(dropdown_widgets)

    if desired > current:
        for _ in range(desired-current):
            dropdown_widgets.append(
                SearchableDropdown(parent=dropdown_frame, 
                                   options=var_options,
                                   row=len(dropdown_widgets)
                                   )
            )

    elif desired < current:
        for _ in range(current - desired):
            w = dropdown_widgets.pop()
            w.dropdown.destroy()

def get_selected_values():
    selected_values = [w.get_value() for w in dropdown_widgets if w.get_value()]
    selected_values = list(dict.fromkeys(selected_values))  
    return selected_values


# Run Commands
def run_explore_multivariate():
    selected = [w.get_value() for w in dropdown_widgets if w.get_value()]
    selected = list(dict.fromkeys(selected))  

    if not selected:
        messagebox.showwarning("No variables selected", 
                               "Select at least 1 variable.")
        return

    explore_multi_variables(*selected)

def run_multivariate_vis():
    selected = [w.get_value() for w in dropdown_widgets if w.get_value()]
    selected = list(dict.fromkeys(selected)) 

    if len(selected) < 2:
        messagebox.showwarning(
            "Not enough variables",
            "Please select at least two variables."
        )
        return

    multivariate_visualizations(*selected)

def run_multivariate_corr():
    selected = [w.get_value() for w in dropdown_widgets if w.get_value()]
    selected = list(dict.fromkeys(selected)) 

    if len(selected) < 2:
        messagebox.showwarning(
            "Not enough variables",
            "Please select at least two variables."
        )
        return

    correlational_analysis(*selected)

def run_gmm():
    selected = [w.get_value() for w in dropdown_widgets if w.get_value()]
    selected = list(dict.fromkeys(selected))

    if len(selected) < 2:
        messagebox.showwarning(
            "Not enough variables",
            "Please select at least two variables."
        )
        return
    gmm_analysis(*selected)

def run_master_gen():
    master_descriptive_csv_generator()
    messagebox.showinfo("Success!", "Master Descriptive CSV Generated")
    return

def run_all_var_gen():
    path = Path(SINGLE_VAR_DESCRIPTIVES_PATH)
    path.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.is_dir() and any(path.iterdir()):
        response = messagebox.askyesno(
            title="Single Variable Descriptives Folder Not Empty",
            message=(
                "The folder already contains files.\n"
                "Continuing will regenerate all single variables.\n\n"
                "Do you want to continue?"
            )
        )
        if not response:
            return
        
    all_single_var_descriptive_csv_generator()
    messagebox.showinfo("Success!", "All Single Variable Descriptive CSVs Generated")
    return


# Variable count spinbox
vc_str_var = tk.StringVar(value='1')

var_count_selection = tk.Spinbox(top_frame, from_=1, to=10, width=5, 
                                 textvariable=vc_str_var, 
                                 command=update_dropdowns
                                 )
var_count_selection.grid(row=2, column=0, columnspan=3, pady=(10, 0))

vc_label = ttk.Label(top_frame, text='Number of Variables?')
vc_label.grid(row=1,column=0, columnspan=3, pady=(10, 0))

vc_str_var.trace_add('write', update_dropdowns)


# Number of variable dropdowns present initially
update_dropdowns()


# Multivariate Buttons
btn_explore_multi_var = tk.Button(top_frame, 
                                  text='Multivariate Exploration',
                            command=run_explore_multivariate)
btn_explore_multi_var.grid(row=0, column=0, padx=5, sticky="nsew")

btn_multi_vis = tk.Button(top_frame, text='Multivariate Visualization',
                           command=run_multivariate_vis)
btn_multi_vis.grid(row=0, column=1, padx=5, sticky="nsew")

btn_multi_corr = tk.Button(top_frame, text='Multivariate Correlation',
                           command=run_multivariate_corr)
btn_multi_corr.grid(row=0, column=2, padx=5, sticky="nsew")

btn_gmm = tk.Button(top_frame, text='GMM Analysis',
                    command=run_gmm)
btn_gmm.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")


# Full-Data Buttons
btn_master_desc = tk.Button(bottom_frame, 
                            text='Master Descriptive CSV Generator',
                            command=run_master_gen)
btn_master_desc.grid(row=0, column=0, padx=5, sticky="nsew")

btn_all_single = tk.Button(bottom_frame, 
                           text='All Single Var Descriptive CSV Generator',
                           command=all_single_var_descriptive_csv_generator)
btn_all_single.grid(row=0, column=1, padx=5, sticky="nsew")

# Weight Adjustments
for i in range(3):
    top_frame.columnconfigure(i, weight=1)


# Start GUI loop
root.mainloop()
