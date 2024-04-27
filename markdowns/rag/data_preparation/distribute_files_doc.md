## Maximize speed of data preparation process

There are many files to process. The following block will evenly distribute all the files across N buckets based on their total byte size.

Then, each bucket will be a dynamic child block so that they can chunk and tokenize the files in their bucket concurrently with the other buckets.