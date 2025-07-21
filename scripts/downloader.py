# How to Run
# pip install colorama # For Windows users, to make ANSI colors work in cmd.exe
# python downloader.py \
#   --username "userid" \
#   --password "1234" \
#   --download-url "https://developer.deepx.ai/?url=2262" \
#   --save-location "download_dir/" # Optional. If omitted, defaults to 'download/' \
#   --expected-version "1.60.2" # Optional: Expected version string to check in the downloaded filename.

import requests
from bs4 import BeautifulSoup
import os
import argparse
import sys
import math
import urllib.parse # For unquoting URL components

# Optional: For Windows compatibility with ANSI escape codes
try:
    import colorama
    colorama.init()
except ImportError:
    pass

# ANSI color codes
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"

def colored_print(message, level="INFO"):
    """Prints a colored log message based on its level."""
    if level == "ERROR":
        sys.stderr.write(f"{COLOR_RED}{message}{COLOR_RESET}\n")
    elif level == "WARNING":
        sys.stdout.write(f"{COLOR_YELLOW}{message}{COLOR_RESET}\n")
    elif level == "INFO":
        sys.stdout.write(f"{COLOR_GREEN}{message}{COLOR_RESET}\n")
    elif level == "DEBUG": # For potential future use
        sys.stdout.write(f"{COLOR_BLUE}{message}{COLOR_RESET}\n")
    else:
        sys.stdout.write(f"{message}\n")
    sys.stdout.flush()
    sys.stderr.flush() # Ensure error messages are flushed

def human_readable_size(size_bytes):
    """Converts a size in bytes to a human-readable format (e.g., KB, MB, GB)."""
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', print_end = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix}{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write(print_end)
        sys.stdout.flush()

def login_and_download_file(username, password, download_url, save_directory, expected_version=None):
    """
    Logs into the DEEPX Developers' page and downloads a file from a specific URL.

    Args:
        username (str): The username or email to log in with.
        password (str): The password to log in with.
        download_url (str): The URL of the file to download.
        save_directory (str): The directory to save the downloaded file to.
        expected_version (str, optional): The version string expected in the downloaded filename.
                                         If not None, checks if the determined filename contains this.

    Returns:
        str or None: The full path to the downloaded file if successful, None otherwise.
    """
    session = requests.Session()
    login_page_url = "https://developer.deepx.ai/"

    # Ensure the save directory exists
    try:
        os.makedirs(save_directory, exist_ok=True)
    except OSError as e:
        colored_print(f"ERROR: Could not create save directory '{save_directory}': {e}", "ERROR")
        return None

    # 1. Send a GET request to the login page to extract CSRF token and other necessary values
    try:
        colored_print("INFO: Extracting login page information...", "INFO")
        response_get = session.get(login_page_url)
        response_get.raise_for_status() # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        colored_print(f"ERROR: Failed to retrieve login page: {e}", "ERROR")
        return None

    soup = BeautifulSoup(response_get.text, 'html.parser')

    csrf_token = None
    wp_http_referer = None

    # Extract CSRF token
    csrf_input = soup.find('input', {'id': 'CSRFToken-wppb'})
    if csrf_input:
        csrf_token = csrf_input.get('value')
    else:
        colored_print("ERROR: CSRF token not found. The website's HTML structure might have changed.", "ERROR")
        return None

    # Extract _wp_http_referer (hidden input with name _wp_http_referer)
    wp_http_referer_input = soup.find('input', {'name': '_wp_http_referer'})
    if wp_http_referer_input:
        wp_http_referer = wp_http_referer_input.get('value')
    else:
        colored_print("ERROR: '_wp_http_referer' value not found. The website's HTML structure might have changed.", "ERROR")
        return None

    if not csrf_token or not wp_http_referer:
        colored_print("ERROR: Failed to extract necessary login information.", "ERROR")
        return None

    # 2. Send a POST request using the extracted information for login
    login_data = {
        'log': username,
        'pwd': password,
        'wp-submit': 'Log In',
        'redirect_to': 'https://developer.deepx.ai/',
        'wppb_login': 'true',
        'wppb_form_location': 'widget',
        'wppb_request_url': 'https://developer.deepx.ai/',
        'wppb_lostpassword_url': '',
        'wppb_redirect_priority': '',
        'wppb_referer_url': 'https://developer.deepx.ai/',
        'CSRFToken-wppb': csrf_token,
        '_wp_http_referer': wp_http_referer,
        'wppb_redirect_check': 'true',
        'rememberme': 'forever'
    }

    try:
        colored_print("INFO: Attempting to log in...", "INFO")
        response_post = session.post(login_page_url, data=login_data, allow_redirects=True)
        response_post.raise_for_status() # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        colored_print(f"ERROR: An error occurred during login request: {e}", "ERROR")
        return None

    # 3. Enhanced Login Success/Failure Check based on URL and content
    if "loginerror=" in response_post.url:
        colored_print("ERROR: Login failed. The provided username or password is invalid. Please check your credentials.", "ERROR")
        return None
    elif "wp-login.php" in response_post.url:
        colored_print("ERROR: Login failed. The server redirected back to the login page unexpectedly. Please check your credentials.", "ERROR")
        return None
    elif "잘못된 비밀번호" in response_post.text or "알 수 없는 사용자" in response_post.text:
        colored_print("ERROR: Login failed. The website indicated an invalid username or password in its content. Please check your credentials.", "ERROR")
        return None
    else:
        colored_print(f"INFO: Login successful! Final URL: {response_post.url}", "INFO")

    # 4. Download the file after successful login with more robust checks and custom progress bar
    full_save_path = None
    try:
        colored_print(f"INFO: Requesting file download from: {download_url}", "INFO")
        file_response = session.get(download_url, stream=True)
        file_response.raise_for_status()

        total_size = int(file_response.headers.get('content-length', 0))
        downloaded_bytes = 0

        # Read first chunk for preliminary checks
        initial_content_chunk = b''
        try:
            initial_content_chunk = next(file_response.iter_content(chunk_size=1024))
        except StopIteration:
            colored_print(f"ERROR: No content received for download from '{download_url}'. File might not exist or be empty.", "ERROR")
            return None
        
        content_preview = initial_content_chunk.decode('utf-8', errors='ignore') 

        denial_message_text = "You are not allowed to access this file."
        if denial_message_text in content_preview or (total_size < 5000 and "<!DOCTYPE html>" in content_preview.lower()):
            colored_print(f"ERROR: File access denied or invalid file content received for '{download_url}'.", "ERROR")
            if denial_message_text in content_preview:
                colored_print(f"Server response indicates: '{denial_message_text}'", "ERROR")
            else:
                colored_print("Server returned an HTML page, possibly an error or redirection, instead of a file.", "ERROR")
            colored_print("Please ensure your account has the necessary permissions to download this file and the URL is correct.", "ERROR")
            return None

        # --- Dynamic Filename Determination Logic ---
        determined_filename = None

        # 1. Try to get filename from Content-Disposition header
        if 'Content-Disposition' in file_response.headers:
            cd = file_response.headers['Content-Disposition']
            if "filename*=" in cd:
                try:
                    parts = cd.split("filename*=")[1].split(";")[0]
                    encoding_and_filename = parts.split("''", 1)
                    if len(encoding_and_filename) == 2:
                        encoding = encoding_and_filename[0].strip().lower()
                        encoded_filename = encoding_and_filename[1].strip('"\'')
                        determined_filename = urllib.parse.unquote(encoded_filename, encoding=encoding if encoding else 'utf-8')
                    else:
                        colored_print(f"WARNING: Unexpected filename* format: {parts}. Trying simple filename= extraction.", "WARNING")
                        if "filename=" in cd:
                             determined_filename = cd.split('filename=')[1].strip('"\'')
                except Exception as e:
                    colored_print(f"WARNING: Error parsing filename* from Content-Disposition: {e}. Falling back to simple filename=.", "WARNING")
                    if "filename=" in cd:
                        determined_filename = cd.split('filename=')[1].strip('"\'')
            elif "filename=" in cd:
                determined_filename = cd.split('filename=')[1].strip('"\'')

        # 2. If no filename from Content-Disposition, try to get it from the final URL path
        if not determined_filename:
            parsed_url = urllib.parse.urlparse(file_response.url)
            path_segments = parsed_url.path.split('/')
            potential_filename_from_url = path_segments[-1] if path_segments else ''

            if potential_filename_from_url and '.' in potential_filename_from_url:
                determined_filename = potential_filename_from_url
            
            if determined_filename and '?' in determined_filename:
                determined_filename = determined_filename.split('?')[0]

        # --- NEW: Error if filename cannot be determined ---
        if not determined_filename:
            colored_print("ERROR: Could not automatically determine the filename from Content-Disposition or URL.", "ERROR")
            colored_print("Please ensure the download URL is valid and the server provides filename information.", "ERROR")
            return None

        # --- NEW: Verify version in filename ---
        if expected_version and expected_version not in determined_filename:
            colored_print(f"ERROR: Downloaded filename '{determined_filename}' does not contain the expected version '{expected_version}'. Aborting.", "ERROR")
            return None

        full_save_path = os.path.join(save_directory, determined_filename)

        colored_print(f"INFO: Saving file as '{determined_filename}' in directory '{save_directory}'.", "INFO")

        # Initialize progress bar
        colored_print(f"INFO: Downloading '{determined_filename}'...", "INFO")
        if total_size > 0:
            print_progress_bar(0, total_size, prefix='Progress:', suffix=f'({human_readable_size(0)}/{human_readable_size(total_size)})', length=50)
        else:
            sys.stdout.write(f'\rProgress: 0 B downloaded...')
            sys.stdout.flush()

        with open(full_save_path, 'wb') as f:
            if initial_content_chunk:
                f.write(initial_content_chunk)
                downloaded_bytes += len(initial_content_chunk)
                if total_size > 0:
                    print_progress_bar(downloaded_bytes, total_size, prefix='Progress:', suffix=f'({human_readable_size(downloaded_bytes)}/{human_readable_size(total_size)})', length=50)
                else:
                    sys.stdout.write(f'\rProgress: {human_readable_size(downloaded_bytes)} downloaded...')
                    sys.stdout.flush()
            
            for chunk in file_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    if total_size > 0:
                        print_progress_bar(downloaded_bytes, total_size, prefix='Progress:', suffix=f'({human_readable_size(downloaded_bytes)}/{human_readable_size(total_size)})', length=50)
                    else:
                        sys.stdout.write(f'\rProgress: {human_readable_size(downloaded_bytes)} downloaded...')
                        sys.stdout.flush()
        
        # Final progress bar update and newline
        if total_size > 0:
            print_progress_bar(total_size, total_size, prefix='Progress:', suffix=f'({human_readable_size(total_size)}/{human_readable_size(total_size)})', length=50, print_end='\n')
        else:
            sys.stdout.write('\n')
            sys.stdout.flush()
        
        final_downloaded_size = os.path.getsize(full_save_path)

        if final_downloaded_size == 0:
            colored_print(f"ERROR: Downloaded file '{full_save_path}' is empty. This often indicates a server-side error or incorrect URL.", "ERROR")
            os.remove(full_save_path)
            return None
        elif total_size > 0 and final_downloaded_size < total_size:
            colored_print(f"WARNING: Downloaded file size ({human_readable_size(final_downloaded_size)}) is less than expected ({human_readable_size(total_size)}). File might be incomplete.", "WARNING")
            os.remove(full_save_path)
            return None
        
        colored_print(f"SUCCESS: File successfully downloaded and saved to '{full_save_path}'. Size: {human_readable_size(final_downloaded_size)}.", "INFO")
        return full_save_path # Return the full path to the downloaded file

    except requests.exceptions.RequestException as e:
        colored_print(f"ERROR: An HTTP/network error occurred during file download: {e}", "ERROR")
        if full_save_path and os.path.exists(full_save_path):
            os.remove(full_save_path)
            colored_print(f"INFO: Removed incomplete file at '{full_save_path}'.", "INFO")
        return None
    except IOError as e:
        colored_print(f"ERROR: An error occurred while saving the file to disk: {e}", "ERROR")
        if full_save_path and os.path.exists(full_save_path):
            os.remove(full_save_path)
            colored_print(f"INFO: Removed incomplete file at '{full_save_path}'.", "INFO")
        return None
    except Exception as e:
        colored_print(f"ERROR: An unexpected error occurred during file download: {e}", "ERROR")
        if full_save_path and os.path.exists(full_save_path):
            os.remove(full_save_path)
            colored_print(f"INFO: Removed incomplete file at '{full_save_path}'.", "INFO")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log in to DEEPX Developers' page and download a file.")
    parser.add_argument('-u', '--username', required=True, help="Username or email for login for 'developer.deepx.ai'.")
    parser.add_argument('-p', '--password', required=True, help="Password for login.")
    parser.add_argument('-d', '--download-url', required=True, help="URL of the file to download (e.g., 'https://developer.deepx.ai/?url=2262').")
    parser.add_argument('-s', '--save-location', default='downloads', help="Directory to save the downloaded file (e.g., 'downloads/'). Defaults to 'downloads'.")
    parser.add_argument('-v', '--expected-version', help="Optional: Expected version string to check in the downloaded filename.")
    args = parser.parse_args()

    # Ensure save_location is treated as a directory
    save_directory = args.save_location
    if not save_directory.endswith(os.sep):
        save_directory += os.sep

    # Pass the expected_version to the download function
    downloaded_file_path = login_and_download_file(
        args.username,
        args.password,
        args.download_url,
        save_directory,
        args.expected_version
    )

    if downloaded_file_path is None:
        colored_print("ERROR: Operation failed. Exiting with a non-zero status code.", "ERROR")
        sys.exit(1)
    else:
        colored_print("INFO: All operations completed successfully.", "INFO")
        sys.exit(0)