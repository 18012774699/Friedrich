import os
import pandas as pd
import xlsxwriter
from Api.file_search import FindFileQuicklyByName
from Api.base import Base


class ExcelIO(Base):
    def __init__(self):
        Base.__init__(self)


def read_excel_data(sheet_data: pd.DataFrame) -> list:
    df = sheet_data[sheet_data['team'] == 'UMA']
    return df['file name'].drop_duplicates().tolist()


class ReadExcel(ExcelIO):
    def __init__(self, excel_path: str):
        ExcelIO.__init__(self)
        self.excel_path = excel_path


class FilterAMU(ReadExcel, FindFileQuicklyByName):
    def __init__(self, excel_path: str, search_path: str):
        ReadExcel.__init__(self, excel_path)
        FindFileQuicklyByName.__init__(self, search_path, ['.c', '.h', '.cpp'])

    def process_by_file_name(self, process_func: callable, excel_filter_algo: callable = read_excel_data, step: int = 10000):
        file_count = 0
        sheet_data = pd.read_excel(io=self.excel_path)
        file_list = excel_filter_algo(sheet_data)
        for file_name in file_list:
            if file_count > step:
                break
            process_func(super().return_path_by_file_name(file_name))
            file_count += 1

    def process_by_file_path(self, process_func: callable, excel_filter_algo: callable = read_excel_data, step: int = 10000):  # 单次扫描上限
        file_count = 0
        sheet_data = pd.read_excel(io=self.excel_path)
        file_list = excel_filter_algo(sheet_data)
        for file_path in file_list:
            if file_count > step:
                break
            (filepath, file_name) = os.path.split(file_path)
            process_func(super().return_path_by_file_name(file_name))
            file_count += 1


# Account Marketing Unit(AMU)
class LoadAMU(ReadExcel):
    def __init__(self, excel_path: str):
        ReadExcel.__init__(self, excel_path)

    def return_amu_list(self, team_name: str):
        sheet_data = pd.read_excel(io=self.excel_path)
        df = sheet_data[sheet_data['资源组'] == team_name]
        return df['File Name'].drop_duplicates().tolist()


class ExcelSheetInfo(Base):
    def __init__(self, sheet_name: str = "Willson",
                 column: list = ['Flie Path', 'Flie Name', 'Line Num', 'Group', 'Content']):
        Base.__init__(self)
        self.sheet_name = sheet_name
        self.count = 0
        self.column = column
        self.worksheet = None


class WriteExcel(ExcelIO):
    excel_book = None
    cell_style = None

    def __init__(self, search_list: list, excel_path: str = "search.xlsx"):
        ExcelIO.__init__(self)
        self.excel_path = excel_path
        self.search_list = search_list
        for index in range(len(self.search_list)):
            self.search_list[index] = ExcelSheetInfo(self.search_list[index])

    def init_excel_table(self):
        WriteExcel.excel_book = xlsxwriter.Workbook('search_res.xlsx')
        # 创建一个样式对象，初始化样式
        WriteExcel.cell_style = [WriteExcel.excel_book.add_format()] * 2
        WriteExcel.cell_style[0].set_bold(True)
        WriteExcel.cell_style[0].set_bg_color('#00FF00')
        WriteExcel.cell_style[0].set_align('center')
        WriteExcel.cell_style[1].set_bold(True)
        WriteExcel.cell_style[1].set_align('center')

        for index in range(len(self.search_list)):
            self.search_list[index].worksheet = WriteExcel.excel_book.add_worksheet(self.search_list[index].sheet_name)
            for i in range(len(self.search_list[index].column)):
                self.search_list[index].worksheet.write(0, i, self.search_list[index].column[i],
                                                        WriteExcel.cell_style[0])
            self.search_list[index].worksheet.set_column('A:A', 40)
            self.search_list[index].worksheet.set_column('B:B', 20)
            self.search_list[index].worksheet.set_column('C:C', 10)
            self.search_list[index].worksheet.set_column('D:D', 10)
            self.search_list[index].worksheet.set_column('E:E', 50)

