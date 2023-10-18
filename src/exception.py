import logging
import sys

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info() # exc_tb contains on which file and on which line exception occurred
    filename = exc_tb.tb_frame.f_code.co_filename
    
    error_message = "Error occurred in {0}, line: {1}, error message: {2}".format(
        filename,
        exc_tb.tb_lineno,
        str(error)
    )
    return error_message
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error=error_message, error_detail=error_detail)
        
    def __str__(self) -> str:
        return self.error_message
    
# if __name__ == "__main__":
#     try:
#         num = 1/0
#     except Exception as ex:
#         logging.info("Divide by Zero")    
#         raise CustomException(ex, sys)