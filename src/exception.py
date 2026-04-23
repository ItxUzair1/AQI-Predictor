import sys

class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message=self.get_detailed_message(error_message,error_details)

    def get_detailed_message(self, error_message, error_details: sys):
        _, _, exc_tb = error_details.exc_info()
        
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            file_name = "Unknown File"
            line_number = "Unknown Line"

        return f"Error in script [{file_name}] at line [{line_number}]: {str(error_message)}"

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return self.error_message