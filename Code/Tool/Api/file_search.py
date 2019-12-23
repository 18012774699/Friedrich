import os
from Api.tool import match_extension
from Api.tool import get_file_name_without_extension
from Api.base import Search


class FindFileQuicklyByName(Search):
    def __init__(self, search_path: str, file_type: list = [".c"]):
        Search.__init__(self, file_type)
        self.search_path = search_path
        self.file_path = None
        self.file_name_without_extension = None

    def return_path_by_file_name(self, file_name: str) -> str:
        for main_dir, subdir, file_name_list in os.walk(self.search_path):
            for filename in file_name_list:
                if not match_extension(filename, self.file_type):
                    continue

                self.file_name_without_extension = get_file_name_without_extension(filename)
                if self.file_name_without_extension == file_name or filename == file_name:
                    self.file_path = main_dir + '/' + filename
                    return self.file_path

        Search.error_info.append("{0}: doesn't exist!".format(file_name) + '\n')
        print("{0}: doesn't exist!".format(file_name))
        return ""

