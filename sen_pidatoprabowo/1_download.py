import argparse
import csv
import sys

from tqdm import tqdm
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR


def get_video_urls(cli_urls: list) -> list:
    print("Masukkan URL video YouTube satu per baris. Kosongkan lalu Enter untuk selesai.")
    urls = []
    while True:
        line = input(f"URL video #{len(urls) + 1}: ").strip()
        if not line:
            break
        urls.append(line)
    return urls

def download_comments(video_urls: list, limit: int, sort_by: int, out_path: str):
    downloader = YoutubeCommentDownloader()
    rows = []

    for video_url in video_urls:
        generator = downloader.get_comments_from_url(video_url, sort_by=sort_by)
        for i, comment in enumerate(tqdm(generator, total=limit, desc=f"Mengunduh: {video_url}")):
            if i >= limit:
                break
            rows.append({
                "video_url": video_url,
                "comment_id": comment.get("cid"),
                "author": comment.get("author"),
                "text": comment.get("text"),
                "likes": comment.get("votes"),
                "heart": comment.get("heart", False),
            })
    if not rows:
        print("No comments found.")
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download komentar YouTube ke CSV")
    parser.add_argument("video_urls", nargs="*", default=[], help="Satu atau lebih URL video YouTube, dipisah spasi")
    parser.add_argument("--limit", type=int, default=5000, help="Jumlah maksimum komentar per video (default: 5000, isi 0 untuk semua)")
    parser.add_argument("--out", type=str, default="Komentar_Youtube.csv", help="Nama file CSV output")
    parser.add_argument("--sort", type=str, choices=["popular", "recent"], default="popular", help="Urutan komentar yang diambil")
    args = parser.parse_args()

    video_urls = get_video_urls(args.video_urls)
    sort_by = SORT_BY_POPULAR if args.sort == "popular" else 1
    limit = None if args.limit == 0 else args.limit
    download_comments(video_urls, limit, sort_by, args.out)