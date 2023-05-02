from qt import QApplication, QComboBox, QMainWindow, QWidget, QVBoxLayout, QStandardItemModel, Qt


# creating checkable combo box class
class CheckableComboBox(QComboBox):
    def __init__(self):
        super(CheckableComboBox, self).__init__()
        self.view().pressed.connect(self.handle_item_pressed)
        self.setModel(QStandardItemModel(self))

    # when any item get pressed
    def handle_item_pressed(self, index):
        # getting which item is pressed
        item = self.model().itemFromIndex(index)
        # make it check if unchecked and vice-versa
        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
        else:
            item.setCheckState(Qt.Checked)
        self.check_items()

    # method called by check_items
    def item_checked(self, index):
        item = self.model().item(index, 0)
        return item.checkState() == Qt.Checked

    # calling method
    def check_items(self):
        checkedItems = []
        for i in range(self.count):
            if self.item_checked(i):
                checkedItems.append(i)
        self.update_labels(checkedItems)

    # method to update the label
    def update_labels(self, item_list):
        item_text = [self.model().item(i, 0).text().split("-")[0].strip()
                     for i in range(self.count)]
        if len(item_list) > 0:
            n = ", ".join([item_text[i] for i in item_list])
            item_text_new = [txt + ' - selected items: ' + n for txt in item_text]
        else:
            item_text_new = item_text
        for i in range(self.count):
            self.setItemText(i, item_text_new[i])
