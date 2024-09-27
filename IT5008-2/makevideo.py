from moviepy.editor import ImageSequenceClip
import os


def main():
    frames = os.listdir("path")
    le = len(frames)
    frame_files = [os.path.join("path", f"{i}.png") for i in range(1, le + 1)]
    print(f"Creating video from {le} frames")
    clip = ImageSequenceClip(frame_files, fps=24)
    clip.write_videofile("output.mp4", codec="libx264", fps=24)
    print("Video created")


main()
