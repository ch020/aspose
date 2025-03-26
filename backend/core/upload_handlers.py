from django.core.files.uploadhandler import FileUploadHandler
import sys

class PrintProgressUploadHandler(FileUploadHandler):
    def __init__(self, request=None):
        super().__init__(request)
        self.total = 0
        self.bytes_received = 0

    def handle_raw_input(
        self,
        input_data,
        META,
        content_length,
        boundary,
        encoding = None,
    ):
        self.total = content_length
        print(f"Receiving upload... Total size: {self.total / 1e6:.2f} MB")

    def receive_data_chunk(self, raw_data, start):
        self.bytes_received += len(raw_data)
        percent = int((self.bytes_received / self.total) * 100)
        print(f"    {percent}% complete", end="\r", flush=True)
        return raw_data

    def file_complete(self, file_size):
        print(f"\nUpload complete ({file_size / 1e6:.2f} MB)\n")
        return None