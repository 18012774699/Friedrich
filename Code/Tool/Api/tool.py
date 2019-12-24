# 基本工具函数
import os
import re


def get_file_name_without_extension(file_name: str) -> str:
    (file_name_without_extension, extension) = os.path.splitext(file_name)
    return file_name_without_extension


def fetch_file_name_by_path(filepath: str) -> str:
    (file_path, file_name) = os.path.split(filepath)
    return file_name


def match_extension(file_name: str, file_type: list) -> bool:
    # (filepath, tempfilename) = os.path.split(file_name)
    (filename, extension) = os.path.splitext(file_name)
    return extension in file_type


def remove_space(string: str) -> str:
    for index in range(len(string)):
        if string[index] != " " and string[index] != "\n":
            return string[index:]


def print_file_valid_num(file_path: str):
    file = open(file_path, 'r', errors='ignore')
    lines = file.readlines()
    in_comment = False  # 多行注释
    code_num = 0
    comment_num = 0
    null_num = 0
    total_num = len(lines)

    for item in lines:
        # 多行注释
        if re.match(r'(\s*)(\/\*)', item):
            comment_num += 1
            if re.search(r'\*\/', item) is None:
                in_comment = True
        elif in_comment:
            if re.search(r'\*\/', item):
                comment_num += 1
                in_comment = False
            else:
                comment_num += 1
        # 单行注释
        elif re.match(r'(\s*)\/\/', item):
            comment_num += 1
        elif re.match(r'(\s*)\n', item):
            null_num += 1
        else:
            code_num += 1

    print("文件: {0}, 总行数: {1}, 有效代码行: {2}".format(file.name, total_num, code_num))
