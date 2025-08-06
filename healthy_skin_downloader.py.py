from bing_image_downloader import downloader
import shutil
from pathlib import Path

query_string = "healthy infant skin"
base_output_dir = Path("data/train")
download_dir = base_output_dir / query_string
final_dir = base_output_dir / "healthy"


downloader.download(
    query_string,
    limit=50,
    output_dir=str(base_output_dir),
    adult_filter_off=True,
    force_replace=False,
    timeout=60
)


final_dir.mkdir(parents=True, exist_ok=True)


for file in download_dir.glob("*.*"):
    shutil.move(str(file), final_dir / file.name)


shutil.rmtree(download_dir)

print(f"Photos transfered to: {final_dir}")
