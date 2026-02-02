import os
import sys
import dill
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Docstring for save_object
    
    :param file_path: Files path to be saved at
    :param obj: 
    """

    try:
        dir_path= os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb')as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        CustomException(e, sys)