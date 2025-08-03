from locust import HttpUser, task, between
import os

class UploadUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def upload_bulk_images(self):
        image_folder = "../data/test/high_demand"
        file_tuples = []
        label_list = []

        for idx, fname in enumerate(os.listdir(image_folder)[:5]):
            file_path = os.path.join(image_folder, fname)
            file_tuples.append(
                ("files", (fname, open(file_path, "rb"), "image/jpeg"))
            )
            label_list.append("low_demand")

        # Send labels in data instead of files
        data = [("labels", label) for label in label_list]

        with self.client.post(
            "/upload-bulk",
            files=file_tuples,
            data=data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                print(f"Failed: {response.status_code} - {response.text}")
                response.failure("Upload failed")