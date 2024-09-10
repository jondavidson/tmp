import ipywidgets as widgets
from IPython.display import display
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'Country': ['USA', 'USA', 'Canada', 'Canada', 'Mexico', 'Mexico'],
    'State': ['California', 'Texas', 'Ontario', 'Quebec', 'Yucatan', 'Jalisco'],
    'City': ['Los Angeles', 'Houston', 'Toronto', 'Montreal', 'Merida', 'Guadalajara'],
    'Population': [10000000, 5000000, 3000000, 4000000, 2000000, 1500000],
    'Area': [500, 700, 600, 800, 450, 350]
})

class TreeViewWidget:
    def __init__(self, df):
        self.df = df
        self.selected_columns = []
        self.filters = {}
        
        # Listbox for available columns
        self.available_columns = widgets.SelectMultiple(
            options=list(df.columns),
            description='Available:',
            rows=5
        )
        
        # Listbox for selected columns (for hierarchy)
        self.selected_columns_box = widgets.SelectMultiple(
            options=[],
            description='Selected:',
            rows=5
        )
        
        # Buttons to move columns between available and selected
        self.add_button = widgets.Button(description='Add →')
        self.remove_button = widgets.Button(description='← Remove')
        self.up_button = widgets.Button(description='↑ Move Up')
        self.down_button = widgets.Button(description='↓ Move Down')
        
        # Button to submit the selection and generate the hierarchy
        self.submit_button = widgets.Button(description='Generate Treeview')
        
        # Slider for size selection (based on numeric columns)
        self.size_column = widgets.Dropdown(
            options=[col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
            description='Size By:'
        )
        self.size_slider = widgets.FloatSlider(
            value=10,
            min=5,
            max=100,
            step=1,
            description='Size Factor:'
        )
        
        # Dropdown for color selection (can be based on any column)
        self.color_column = widgets.Dropdown(
            options=list(df.columns),
            description='Color By:'
        )
        
        # Color picker for default node color
        self.color_picker = widgets.ColorPicker(
            value='#ff0000',
            description='Default Color:'
        )
        
        # Layout for the buttons
        button_layout = widgets.VBox([self.add_button, self.remove_button, self.up_button, self.down_button, self.submit_button])
        
        # Layout for size and color selection
        size_color_layout = widgets.VBox([self.size_column, self.size_slider, self.color_column, self.color_picker])
        
        # Event listeners for the buttons
        self.add_button.on_click(self.add_columns)
        self.remove_button.on_click(self.remove_columns)
        self.up_button.on_click(self.move_up)
        self.down_button.on_click(self.move_down)
        self.submit_button.on_click(self.on_submit)
        
        # Layout for the overall widget
        self.container = widgets.HBox([self.available_columns, button_layout, self.selected_columns_box, size_color_layout])
        display(self.container)

        # Accordion to hold the filters
        self.filter_accordion = widgets.Accordion(children=[], titles=[])
        display(self.filter_accordion)
    
    def add_columns(self, button):
        # Move selected columns from the available list to the selected list
        for col in self.available_columns.value:
            if col not in self.selected_columns:
                self.selected_columns.append(col)
                self.add_filter(col)
        
        self.update_selected_listbox()

    def add_filter(self, col):
        """Add filter options for a selected column."""
        # Create a filter accordion item (dropdown or range filter)
        if pd.api.types.is_numeric_dtype(self.df[col]):
            filter_widget = widgets.FloatRangeSlider(
                value=[self.df[col].min(), self.df[col].max()],
                min=self.df[col].min(),
                max=self.df[col].max(),
                step=(self.df[col].max() - self.df[col].min()) / 100,
                description=f'{col} Filter:'
            )
        else:
            filter_widget = widgets.SelectMultiple(
                options=self.df[col].unique(),
                description=f'{col} Filter:'
            )
        
        self.filters[col] = filter_widget
        self.filter_accordion.children += (filter_widget,)
        self.filter_accordion.set_title(len(self.filter_accordion.children) - 1, f'Filter {col}')
    
    def remove_columns(self, button):
        # Remove selected columns from the selected list
        for col in self.selected_columns_box.value:
            if col in self.selected_columns:
                self.selected_columns.remove(col)
                self.remove_filter(col)
        
        self.update_selected_listbox()

    def remove_filter(self, col):
        """Remove filter from the accordion."""
        index = list(self.filters.keys()).index(col)
        self.filters.pop(col)
        self.filter_accordion.children = tuple(child for i, child in enumerate(self.filter_accordion.children) if i != index)

    def move_up(self, button):
        # Move the selected column up in the selected list
        if self.selected_columns_box.value:
            col = self.selected_columns_box.value[0]
            idx = self.selected_columns.index(col)
            if idx > 0:
                self.selected_columns[idx], self.selected_columns[idx - 1] = self.selected_columns[idx - 1], self.selected_columns[idx]
            self.update_selected_listbox()

    def move_down(self, button):
        # Move the selected column down in the selected list
        if self.selected_columns_box.value:
            col = self.selected_columns_box.value[0]
            idx = self.selected_columns.index(col)
            if idx < len(self.selected_columns) - 1:
                self.selected_columns[idx], self.selected_columns[idx + 1] = self.selected_columns[idx + 1], self.selected_columns[idx]
            self.update_selected_listbox()

    def update_selected_listbox(self):
        # Update the selected columns listbox
        self.selected_columns_box.options = self.selected_columns
    
    def on_submit(self, button):
        # On submission, print the selected columns (hierarchy)
        print("Selected Columns for Treeview Hierarchy:", self.selected_columns)
        
        # Size and color settings
        size_by = self.size_column.value
        size_factor = self.size_slider.value
        color_by = self.color_column.value
        default_color = self.color_picker.value
        
        print(f"Size by: {size_by} (Factor: {size_factor})")
        print(f"Color by: {color_by}, Default color: {default_color}")
        
        # Filtered DataFrame
        filtered_df = self.df.copy()
        for col, filter_widget in self.filters.items():
            if isinstance(filter_widget, widgets.FloatRangeSlider):
                filtered_df = filtered_df[(filtered_df[col] >= filter_widget.value[0]) & (filtered_df[col] <= filter_widget.value[1])]
            elif isinstance(filter_widget, widgets.SelectMultiple):
                filtered_df = filtered_df[filtered_df[col].isin(filter_widget.value)]
        
        # Example: grouping the DataFrame and applying size/color logic
        if self.selected_columns:
            grouped = filtered_df.groupby(self.selected_columns).size().reset_index(name='count')
            grouped['Size'] = grouped[size_by] * size_factor if size_by in grouped.columns else size_factor
            grouped['Color'] = grouped[color_by] if color_by in grouped.columns else default_color
            
            print(grouped)

# Usage
treeview_widget = TreeViewWidget(df)
