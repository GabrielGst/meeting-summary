import argparse
from .transcriber import transcribe_video, summarize_existing_file

def main():
    parser = argparse.ArgumentParser(description="Locally transcribe and summarize French media streams using Option B Sequential VRAM mapping.")
    
    parser.add_argument("url", nargs="?", default=None, help="The direct video/audio URL to download and process")
    parser.add_argument("-b", "--browser", default="chrome", choices=["chrome", "firefox", "edge", "safari"],
                        help="Browser preference (default: chrome)")
    parser.add_argument("-o", "--output", default="transcript.md", 
                        help="Path to save the final output markdown file (default: transcript.md)")
    parser.add_argument("-t", "--token", default=None,
                        help="Manually copied Crowdcast _crowdcast_session cookie string token")
    
    # The flag pointing to our exception path
    parser.add_argument("--from_file", default=None,
                        help="Exception Case: Path to an existing .md transcription file to instantly summarize offline")
    
    args = parser.parse_args()
    
    if args.from_file:
        # Executes only the LLM logic on a pre-existing file without touching Whisper
        summarize_existing_file(args.from_file, args.output)
    elif args.url:
        # Executes full sequential download -> transcribe -> clear VRAM -> summary pipeline
        transcribe_video(args.url, args.browser, args.output, session_cookie=args.token)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
