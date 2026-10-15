# Train Rook with your own corrections

1. Open the Rook chat page and talk to him.
2. Under a reply, choose **Correct this reply**, write the answer you wanted, then choose **Save correction**.
3. When you have a few examples, choose **Export training file**. This downloads `rook-user-corrections.jsonl`.
4. In this repository, open the `training` folder on the `main` branch. Use **Add file → Upload files** and upload the downloaded file.
5. Rename the uploaded file to exactly `user_corrections.jsonl` if needed, then commit it to `main`.
6. The training workflow starts automatically because the file is under `training/`. It adds your examples to the training split only. Validation and test examples remain separate.

The corrections stay in your browser until you export them. They are not sent to a server automatically. The exported file contains the prompt and the answer you wrote. Review it before committing because it becomes part of the repository.

This fine-tunes Rook's existing randomly initialized model. It does not use a pretrained model or an external AI to write replies. Training can take hours on free CPU runners, and adding examples does not guarantee an immediate improvement. Check the training report and sampled-reply test before treating a new checkpoint as better.
