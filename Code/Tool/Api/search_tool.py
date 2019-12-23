import xlsxwriter
from Api.base import FileProc


class SearchTool(FileProc):
    def __init__(self, match_algorithm: callable, check_in_function: bool, need_in_function: bool, file_type: list):
        FileProc.__init__(self, check_in_function, need_in_function, file_type)
        self.match_algorithm = match_algorithm
        self.file_path = ""
        self.excel_book = xlsxwriter.Workbook('search_res.xlsx')
        self.cell_style = None

    def file_proc(self, all_lines: list, index: int):
        self.match_algorithm(self.file_path, self.file_name, all_lines, index)

    def check_file(self, file_path: str):
        self.file_path = file_path
        super().check_file(file_path)
