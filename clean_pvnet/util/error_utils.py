from typing import List


def get_type_error_message(
    func_name: str, acceptable_types: List[type],
    unacceptable_type: type, param_name: str = None
):
    acceptable_types_str = ""
    for i, acceptable_type in enumerate(acceptable_types):
        if i == 0:
            acceptable_types_str += f'\t{acceptable_type.__name__}'
        else:
            acceptable_types_str += f'\n\t{acceptable_type.__name__}'
    param_explanation_str = f' for parameter {param_name}' \
        if param_name is not None else ''
    print_str = \
    f"""
    {func_name} does not accept:
        {unacceptable_type.__name__}{param_explanation_str}.
    The following types are acceptable:
    {acceptable_types_str}
    """
    return print_str
