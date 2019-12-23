from Api.base import FileProc
import re


class ReplaceTool(FileProc):
    def __init__(self, replace_algorithm: callable, check_in_function: bool, need_in_function: bool, file_type: list):
        FileProc.__init__(self, check_in_function, need_in_function, file_type)
        self.replace_algorithm = replace_algorithm
        self.file_be_changed = False
        self.all_lines = None
        self.modified_function = []

    def file_proc(self, all_lines: list, index: int):
        self.all_lines, temp = self.replace_algorithm(all_lines, index)
        self.file_be_changed = self.file_be_changed or temp
        # 记录修改函数名
        if temp and self.in_function:
            num = 1
            while not re.match(r'([A-Z][A-Z_0-9]+)(\s+)([A-Z][a-z_A-Z0-9]+)\(', all_lines[index - num]):
                num += 1
            target = re.search(r'([A-Z][a-z_A-Z0-9]+)\(', all_lines[index - num]).group()  # 函数定义行
            self.modified_function.append(target[:-1] + '\n')

    def check_file(self, file_path: str):
        self.file_be_changed = False

        super().check_file(file_path)

        if self.file_be_changed:
            print(self.file_name)
            content = ''.join(self.all_lines)
            write_file = open(file_path, 'w')
            write_file.write(content)
            write_file.close()

        self.all_lines = []

    def log_modified_function(self):
        if len(self.modified_function) != 0:
            self.modified_function = list(set(self.modified_function))
            modified_function = open("modified_function.txt", 'w')
            content = ''.join(self.modified_function)
            modified_function.write(content)
            modified_function.close()

