import argparse
import os

from PIL import Image, ImageDraw, ImageFont, ImageOps


def get_images_from_folder(folder):
    """
    Get a dictionary of images from the folder.
    The keys are tuples (id, iteration) and the values are the image file paths.
    """
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            id, iteration = filename.split(".")[0].split("_")
            images[(id, int(iteration))] = os.path.join(folder, filename)
    return images


def add_border(image, border_size=1, border_color=(0, 0, 0)):
    """
    Add a border to an image.
    """
    return ImageOps.expand(image, border=border_size, fill=border_color)


def add_text(image, text, position, font_size=20):
    """
    Add text to an image at the specified position.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
    text_position = (position[0] - text_size[0] // 2, position[1] - text_size[1] // 2)
    draw.text(text_position, text, font=font, fill=(0, 0, 0))
    return image


def combine_images_horizontally(image_paths, folder_names, padding=10, border_size=1):
    """
    Combine images horizontally with borders, padding, and folder names.
    """
    images = [add_border(Image.open(path), border_size) for path in image_paths]
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths) + (len(images) - 1) * padding
    max_height = max(heights) + 20  # Extra space for folder name text

    combined_image = Image.new("RGB", (total_width, max_height), (255, 255, 255))

    x_offset = 0
    for img, folder_name in zip(images, folder_names):
        combined_image.paste(
            img, (x_offset, 20)
        )  # Paste image with 20px offset for text
        combined_image = add_text(
            combined_image, folder_name, (x_offset + img.width // 2, 10), 20
        )
        x_offset += img.width + padding

    return combined_image


def combine_images_vertically(images, id_text, padding=10, border_size=1):
    """
    Combine images vertically with borders, padding, and id text at the bottom.
    """
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    total_height = (
        sum(heights) + (len(images) - 1) * padding + 20
    )  # Extra space for id text

    combined_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))

    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height + padding

    combined_image = add_text(
        combined_image, id_text, (max_width // 2, y_offset - padding + 10), 20
    )

    return combined_image


def process_folders(input_folders, output_folder, padding=10, border_size=1):
    """
    Process images from input folders and write combined images to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_images = [get_images_from_folder(folder) for folder in input_folders]
    ids = set(id for images in all_images for id, _ in images.keys())

    for id in ids:
        combined_iterations = []
        for iteration in sorted(
            set(
                iteration
                for images in all_images
                for (img_id, iteration) in images.keys()
                if img_id == id
            )
        ):
            images_to_combine = [
                images[(id, iteration)]
                for images in all_images
                if (id, iteration) in images
            ]
            folder_names = [
                folder.split("/")[-2]
                for folder, images in zip(input_folders, all_images)
                if (id, iteration) in images
            ]
            combined_image = combine_images_horizontally(
                images_to_combine, folder_names, padding, border_size
            )
            combined_iterations.append(combined_image)

        final_image = combine_images_vertically(
            combined_iterations, id, padding, border_size
        )
        final_image.save(os.path.join(output_folder, f"{id}.png"))


def main():
    parser = argparse.ArgumentParser(
        description="Combine images from multiple folders."
    )
    parser.add_argument("--input_folders", nargs="+", help="List of input folders")
    parser.add_argument("--output_folder", help="Output folder")
    parser.add_argument(
        "--padding", type=int, default=10, help="Padding between images"
    )
    parser.add_argument(
        "--border_size", type=int, default=1, help="Border size for images"
    )

    args = parser.parse_args()

    process_folders(
        args.input_folders, args.output_folder, args.padding, args.border_size
    )


if __name__ == "__main__":
    main()
