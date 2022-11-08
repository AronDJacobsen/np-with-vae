from torch.utils.tensorboard import SummaryWriter
import sys
import os

class Logger():
    def __init__(self, directory, comment="", write=False):
        self.dir = directory
        self.write = write
        if self.write:
            self.BoardWriter = SummaryWriter(comment=comment)
            self.dir = self.BoardWriter.log_dir
            self.log(f"Logs from {self.dir}\n{' '.join(sys.argv)}\n")

    def write_to_board(self, name, scalars, index=0):
        self.log(f"{name} at {index}: {str(scalars)}")
        if self.write:
            for key, value in scalars.items():
                self.BoardWriter.add_scalar(f"{name}/{key}", value, index)

    def log(self, message):
        if self.write:
            with open(os.path.join(self.dir, "logs.txt"), "a") as logs:
                logs.write(str(message) + "\n")


if __name__ == '__main__':
    l = Logger(directory='runs', comment="_HI_VAE", write=True)


