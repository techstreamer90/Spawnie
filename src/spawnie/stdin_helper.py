"""Helper to send initial prompt then forward stdin to a subprocess."""

import subprocess
import sys
import threading


def forward_stdin(proc):
    """Forward stdin to subprocess."""
    try:
        for line in sys.stdin:
            if proc.poll() is not None:
                break
            proc.stdin.write(line)
            proc.stdin.flush()
    except (BrokenPipeError, OSError):
        pass


def main():
    if len(sys.argv) < 3:
        print("Usage: stdin_helper.py <prompt_file> <command> [args...]", file=sys.stderr)
        sys.exit(1)

    prompt_file = sys.argv[1]
    command = sys.argv[2:]

    # Read initial prompt
    with open(prompt_file, "r", encoding="utf-8") as f:
        initial_prompt = f.read()

    # Start the subprocess
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    # Send initial prompt
    try:
        proc.stdin.write(initial_prompt + "\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        pass

    # Forward stdin in a thread
    stdin_thread = threading.Thread(target=forward_stdin, args=(proc,), daemon=True)
    stdin_thread.start()

    # Wait for subprocess to complete
    proc.wait()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
