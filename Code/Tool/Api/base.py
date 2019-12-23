# 基类
import os
import re
from Api.tool import match_extension
from Api.tool import fetch_file_name_by_path


class Base:
    error_info = []  # 错误信息

    def __init__(self):
        pass

    def __del__(self):
        pass

    @staticmethod
    def output_error_info():
        if len(Base.error_info) != 0:
            logfile = open("error.txt", 'w')
            content = ''.join(Base.error_info)
            logfile.write(content)
            logfile.close()


class Search(Base):
    def __init__(self, file_type: list = None):
        Base.__init__(self)
        self.file_type = [".c", ".h", ".cpp"] if (file_type is None) else file_type


class FileProc(Search):
    def __init__(self, check_in_function: bool = False, need_in_function: bool = True, file_type: list = None):
        Search.__init__(self, file_type)
        self.check_in_function = check_in_function
        self.need_in_function = need_in_function
        self.file_name = None
        self.in_function = False
        # self.total_count = 0

    def file_proc(self, all_lines: list, index: int):
        pass

    def check_file(self, file_path: str):
        self.file_name = fetch_file_name_by_path(file_path)
        read_file = None
        try:
            read_file = open(file_path, 'r', errors='ignore')
            all_lines = read_file.readlines()
        except FileNotFoundError:
            Base.error_info.append("{0}: doesn't exist!".format(self.file_name) + '\n')
            print("{0}: doesn't exist!".format(self.file_name))
            return
        except UnicodeDecodeError:
            read_file.close()
            Base.error_info.append("{0}: read fail!".format(self.file_name) + '\n')
            print("{0}: read fail!".format(self.file_name))
            return

        self.in_function = False
        for index in range(0, len(all_lines)):
            if re.match(r'{', all_lines[index]) and not self.in_function:
                self.in_function = True
            elif self.in_function and re.match(r'}', all_lines[index]):
                self.in_function = False

            if self.check_in_function and self.in_function == self.need_in_function:
                self.file_proc(all_lines, index)
            else:
                self.file_proc(all_lines, index)

    def check_dir(self, search_path: str, filter_amu: list = None):
        for main_dir, subdir, file_name_list in os.walk(search_path):
            # print("main_dir:", main_dir)  # 当前主目录
            # print("subdir:", subdir)  # 当前主目录下的所有目录
            # print("file_name_list:", file_name_list)  # 当前主目录下的所有文件
            for filename in file_name_list:
                if not match_extension(filename, self.file_type):
                    continue
                if filter_amu is not None and filename not in filter_amu:
                    continue
                file_path = main_dir + '/' + filename
                self.check_file(file_path)

