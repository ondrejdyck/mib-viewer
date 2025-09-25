"""
EMD File Inspector Widget - Displays complete file structure and metadata
"""

import json
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget,
                             QTreeWidgetItem, QTextEdit, QSplitter, QLabel,
                             QPushButton, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont

try:
    from ..io.mib_loader import walk_emd_structure
except ImportError:
    from mib_viewer.io.mib_loader import walk_emd_structure


class EMDFileInspector(QWidget):
    """Widget for inspecting EMD file structure and metadata"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_file = None
        self.file_structure = None

        self.setup_ui()
        self.setup_icons()

    def setup_ui(self):
        """Set up the user interface"""
        layout = QVBoxLayout()

        # Header with file info (fixed height, non-resizable)
        self.header_label = QLabel("No EMD file loaded")
        self.header_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.header_label.setFixedHeight(30)  # Fixed height for filename display
        self.header_label.setStyleSheet("padding: 5px; border-bottom: 1px solid #ccc;")
        layout.addWidget(self.header_label)

        # Main splitter: tree on left, metadata on right (takes remaining space)
        splitter = QSplitter(Qt.Horizontal)

        # Left side - Tree widget
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Name", "Type", "Info"])
        self.tree_widget.itemClicked.connect(self.on_item_clicked)
        self.tree_widget.setMinimumWidth(300)
        splitter.addWidget(self.tree_widget)

        # Right side - Metadata display
        self.metadata_panel = QTextEdit()
        self.metadata_panel.setReadOnly(True)
        self.metadata_panel.setFont(QFont("Consolas", 9))
        self.metadata_panel.setMinimumWidth(400)
        self.metadata_panel.setPlainText("Click an item in the tree to view its metadata")
        splitter.addWidget(self.metadata_panel)

        # Set splitter proportions
        splitter.setSizes([350, 500])

        layout.addWidget(splitter)

        # Bottom buttons
        button_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_structure)
        self.refresh_btn.setEnabled(False)
        button_layout.addWidget(self.refresh_btn)

        button_layout.addStretch()

        self.export_btn = QPushButton("Export Structure")
        self.export_btn.clicked.connect(self.export_structure)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def setup_icons(self):
        """Set up icons for different item types"""
        # Note: Using text-based icons for simplicity
        # In a full implementation, you'd use actual icon files
        self.icons = {
            'file': '📄',
            'group': '📁',
            'dataset': '📊',
            'attribute': '🏷️'
        }

    def load_file_structure(self, filename):
        """Load and display EMD file structure"""
        try:
            self.current_file = filename
            self.file_structure = walk_emd_structure(filename)

            self.populate_tree()
            self.update_header()
            self.display_file_summary()

            self.refresh_btn.setEnabled(True)
            self.export_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file structure:\n{str(e)}")

    def populate_tree(self):
        """Populate the tree widget with file structure"""
        if not self.file_structure:
            return

        self.tree_widget.clear()

        # Create root item (file)
        file_info = self.file_structure['file_info']
        root_item = QTreeWidgetItem([
            file_info['filename'],
            "EMD File",
            f"{file_info['filesize_mb']:.1f} MB"
        ])
        root_item.setData(0, Qt.UserRole, {'type': 'file', 'data': file_info})
        self.tree_widget.addTopLevelItem(root_item)

        # Add all items from flat list
        item_map = {'/': root_item}  # Map path to tree item

        for item_data in self.file_structure['flat_items']:
            if item_data['path'] == '/':
                continue  # Skip root, already added

            # Find parent item
            parent_path = '/'.join(item_data['path'].split('/')[:-1]) or '/'
            parent_item = item_map.get(parent_path, root_item)

            # Create tree item
            tree_item = self.create_tree_item(item_data, parent_item)
            item_map[item_data['path']] = tree_item

        # Expand root and first level
        root_item.setExpanded(True)
        for i in range(root_item.childCount()):
            child = root_item.child(i)
            child.setExpanded(True)

    def create_tree_item(self, item_data, parent_item):
        """Create a tree widget item from item data"""
        item_type = item_data['type']
        name = item_data['name']

        # Format info column based on type
        if item_type == 'group':
            info = f"{len(item_data['children'])} items"
        elif item_type == 'dataset':
            shape_str = '×'.join(map(str, item_data['shape']))
            info = f"({shape_str}) {item_data['dtype']}"
        else:
            info = ""

        # Create tree item
        tree_item = QTreeWidgetItem([
            f"{self.icons.get(item_type, '▫')} {name}",
            item_type.capitalize(),
            info
        ])

        # Store full data for metadata display
        tree_item.setData(0, Qt.UserRole, {'type': item_type, 'data': item_data})

        parent_item.addChild(tree_item)
        return tree_item

    def update_header(self):
        """Update the header label with file info"""
        if not self.file_structure:
            self.header_label.setText("No EMD file loaded")
            return

        file_info = self.file_structure['file_info']
        header_text = (f"File: {file_info['filename']} "
                      f"({file_info['filesize_mb']:.1f} MB) - "
                      f"{len(self.file_structure['flat_items'])} items")
        self.header_label.setText(header_text)

    def on_item_clicked(self, item, column):
        """Handle tree item click - display metadata"""
        item_info = item.data(0, Qt.UserRole)
        if not item_info:
            return

        item_type = item_info['type']
        item_data = item_info['data']

        # Format metadata based on item type
        if item_type == 'file':
            self.display_file_metadata(item_data)
        elif item_type == 'group':
            self.display_group_metadata(item_data)
        elif item_type == 'dataset':
            self.display_dataset_metadata(item_data)

    def display_file_summary(self):
        """Display overall file summary in metadata panel"""
        if not self.file_structure:
            return

        file_info = self.file_structure['file_info']

        summary = f"""EMD FILE STRUCTURE OVERVIEW
{'=' * 50}

File: {file_info['filename']}
Path: {file_info['filepath']}
Size: {file_info['filesize_mb']:.2f} MB ({file_info['filesize_bytes']:,} bytes)
HDF5 Version: {file_info.get('hdf5_version', 'unknown')}

Total Items: {len(self.file_structure['flat_items'])}

File Attributes:
{self.format_attributes_section(file_info.get('attributes', {}))}

Structure Summary:
- Groups: {len([item for item in self.file_structure['flat_items'] if item['type'] == 'group'])}
- Datasets: {len([item for item in self.file_structure['flat_items'] if item['type'] == 'dataset'])}

Click on any item in the tree to view detailed metadata.
"""
        self.metadata_panel.setPlainText(summary)

    def display_file_metadata(self, file_data):
        """Display file-level metadata"""
        metadata_text = f"""FILE METADATA
{'=' * 30}

Filename: {file_data['filename']}
Full Path: {file_data['filepath']}
File Size: {file_data['filesize_mb']:.2f} MB ({file_data['filesize_bytes']:,} bytes)
HDF5 Version: {file_data.get('hdf5_version', 'unknown')}

FILE ATTRIBUTES:
{self.format_attributes_section(file_data.get('attributes', {}))}
"""
        self.metadata_panel.setPlainText(metadata_text)

    def display_group_metadata(self, group_data):
        """Display group metadata"""
        metadata_text = f"""GROUP: {group_data['name']}
{'=' * (8 + len(group_data['name']))}

Path: {group_data['path']}
Type: HDF5 Group
Children: {len(group_data['children'])}

CHILDREN:
{chr(10).join(f"  - {child}" for child in group_data['children'])}

GROUP ATTRIBUTES:
{self.format_attributes_section(group_data['attributes'])}
"""
        self.metadata_panel.setPlainText(metadata_text)

    def display_dataset_metadata(self, dataset_data):
        """Display dataset metadata"""
        # Calculate memory usage
        size_mb = dataset_data['size_mb']
        if size_mb < 1:
            size_str = f"{dataset_data['size_bytes']:,} bytes"
        else:
            size_str = f"{size_mb:.2f} MB"

        metadata_text = f"""DATASET: {dataset_data['name']}
{'=' * (10 + len(dataset_data['name']))}

Path: {dataset_data['path']}
Shape: {dataset_data['shape']}
Data Type: {dataset_data['dtype']}
Memory Size: {size_str}
Chunks: {dataset_data['chunks']}
Compression: {dataset_data['compression'] or 'None'}

"""

        # Add statistics if available
        if 'statistics' in dataset_data:
            stats = dataset_data['statistics']
            metadata_text += f"""DATA STATISTICS:
Min: {stats['min']:.6f}
Max: {stats['max']:.6f}
Mean: {stats['mean']:.6f}
Std Dev: {stats['std']:.6f}

"""

        metadata_text += f"""DATASET ATTRIBUTES:
{self.format_attributes_section(dataset_data['attributes'])}
"""

        self.metadata_panel.setPlainText(metadata_text)

    def format_attribute_value(self, key, value):
        """Smart formatting for attribute values, especially JSON strings"""
        if not isinstance(value, str):
            return f"{key}: {value}"

        # Try to parse as JSON and pretty-print
        try:
            parsed = json.loads(value)
            return f"{key}:\n{json.dumps(parsed, indent=2, default=str)}"
        except (json.JSONDecodeError, TypeError):
            # Not JSON, return as regular string
            return f"{key}: {value}"

    def format_attributes_section(self, attributes):
        """Format attributes dictionary with smart JSON handling"""
        if not attributes:
            return "No attributes"

        formatted_items = []
        for key, value in attributes.items():
            formatted_items.append(self.format_attribute_value(key, value))

        return "\n\n".join(formatted_items)

    def refresh_structure(self):
        """Refresh the file structure"""
        if self.current_file:
            self.load_file_structure(self.current_file)

    def export_structure(self):
        """Export the complete file structure to JSON"""
        if not self.file_structure:
            return

        from PyQt5.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export File Structure",
            f"{self.file_structure['file_info']['filename']}_structure.json",
            "JSON files (*.json);;All files (*.*)"
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.file_structure, f, indent=2, default=str)
                QMessageBox.information(self, "Success", f"Structure exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export structure:\n{str(e)}")

    def clear(self):
        """Clear the inspector"""
        self.current_file = None
        self.file_structure = None
        self.tree_widget.clear()
        self.metadata_panel.setPlainText("No EMD file loaded")
        self.header_label.setText("No EMD file loaded")
        self.refresh_btn.setEnabled(False)
        self.export_btn.setEnabled(False)