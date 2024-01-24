import tkinter as tk
import requests
	
# Function to get a new Sudoku puzzle from the API
def get_new_sudoku():
	url = "https://sudoku-api.vercel.app/api/dosuku"
	response = requests.get(url)

	# If the request was successful, parse the JSON response
	if response.status_code == 200:
		return response.json()
	else:
		print(f"Request failed with status code {response.status_code}")
		return None

# Function to check the filled values
def check_values():
	global sudoku_dict, sudoku_grid, sudoku_solution, entries  # Declare these variables as global so we can modify them
	for i in range(9):
		for j in range(9):
			current_value = entries[i][j].get()
			# If the cell was filled by the user
			if current_value:
				# If the value is correct, color it green
				if int(current_value) == sudoku_solution[i][j]:
					entries[i][j].config(fg='green')
				# If the value is incorrect, color it red and change it to the correct value
				else:
					entries[i][j].config(fg='red')
					entries[i][j].delete(0, tk.END)
					entries[i][j].insert(0, str(sudoku_solution[i][j]))

# Function to fill in the solution
def fill_solution():
	global sudoku_dict, sudoku_grid, sudoku_solution, entries  # Declare these variables as global so we can modify them
	for i in range(9):
		for j in range(9):
			entries[i][j].config(state='normal')  # Make the cell changeable
			current_value = entries[i][j].get()
			# If the cell was filled by the user
			if current_value:
				# If the value is correct, color it green
				if int(current_value) == sudoku_solution[i][j]:
					entries[i][j].config(fg='green')
				# If the value is incorrect, color it red
				else:
					entries[i][j].config(fg='red')
			# If the cell was not filled by the user, fill in the correct value
			else:
				entries[i][j].delete(0, tk.END)
				entries[i][j].insert(0, str(sudoku_solution[i][j]))
			entries[i][j].config(state='readonly')  # Make the cell unchangeable
					
# Function to clear the grid and get a new Sudoku puzzle
def clear_grid():
	global sudoku_dict, sudoku_grid, sudoku_solution, entries  # Declare these variables as global so we can modify them
	sudoku_dict = get_new_sudoku()  # Get a new Sudoku puzzle
	sudoku_grid = sudoku_dict['newboard']['grids'][0]['value']
	sudoku_solution = sudoku_dict['newboard']['grids'][0]['solution']
	for i in range(9):
		for j in range(9):
			entries[i][j].config(state='normal')  # Make the cell changeable
			entries[i][j].delete(0, tk.END)
			entries[i][j].config(fg='black')  # Reset the text color to black
			if sudoku_grid[i][j] != 0:  # Only fill in the cells that are not empty
				entries[i][j].insert(0, str(sudoku_grid[i][j]))
				entries[i][j].config(state='readonly')  # Make the cell unchangeable

def open_sudoku_window():
	global sudoku_dict, sudoku_grid, sudoku_solution, entries  # Declare these variables as global so we can modify them
	# Initial Sudoku dictionary
	sudoku_dict = get_new_sudoku()

	# Extract the Sudoku grid and solution
	sudoku_grid = sudoku_dict['newboard']['grids'][0]['value']
	sudoku_solution = sudoku_dict['newboard']['grids'][0]['solution']

	# Create a new tkinter window
	root = tk.Tk()

	# Set the window size
	root.geometry("505x540")  # Width x Height

	# Create a 9x9 grid of Entry widgets
	entries = [[tk.Entry(root, width=2, font=('Arial', 35)) for j in range(9)] for i in range(9)]

	# Fill in the initial values
	for i in range(9):
		for j in range(9):
			value = sudoku_grid[i][j]
			if value != 0:  # Only fill in the cells that are not empty
				entries[i][j].insert(0, str(value))

	# Grid the entries
	for i in range(9):
		for j in range(9):
			entries[i][j].grid(row=i, column=j)
	
	# Create the Check, Solution and Reset buttons
	check_button = tk.Button(root, text="Check", command=check_values, bg='light yellow')
	solution_button = tk.Button(root, text="Solution", command=fill_solution, bg='light blue')
	reset_button = tk.Button(root, text="Reset", command=clear_grid, bg='light green')

	# Grid the buttons
	check_button.grid(row=9, column=0, columnspan=3, sticky="ew")
	solution_button.grid(row=9, column=3, columnspan=3, sticky="ew")
	reset_button.grid(row=9, column=6, columnspan=3, sticky="ew")

	# Run the tkinter event loop
	root.mainloop()