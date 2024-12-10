def convert_to_abs_path(abs_path: str, path: str) -> str:
    """
    Converts relative path to absolute path.
    :param abs_path: an absolute path to current folder
    :param path: absolute or relative path to file
    :return: absolute path to file
    """
    if (
        path[0] == "\\"
        or path[0] == "/"
        or -1 < path.find(":") < max(path.find("/"), path.find("\\"))
    ):
        return path
    else:
        return abs_path + path
