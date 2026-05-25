from pypdf import PdfReader

def load_pdf(file_path):
    # open a file like a book
    reader = PdfReader(file_path)

# Initialize an empty string to hold the combined text from all pages
    all_text = ""

# flip through each page and extract the text
    for page_num,page in enumerate(reader.pages):
        page_text = page.extract_text()

        # if the page was not empty, add its text to the combined string
        if page_text:  
            all_text += page_text + "\n" 

    return all_text

def chunk_text(text, chunk_size=200,overlap=50):

    # empty list to hold the final chunks
    chunks = []

    # set start point at the beginning of the text
    start = 0

    # loop until we reach the end of the text
    while start < len(text):
        # fgure out where the end of the chunk should be
        end = start + chunk_size
        
        # grab the chunk of text
        chunk  = text[start:end]

        # add the chunk to our list of chunks
        chunks.append(chunk)

        # move the start point forward by the chunk size minus the overlap
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":

# load the PDF and extract the text
    pdf_path = r"C:\Users\prash\Desktop\govprep\data\ncert_polity_ch01.pdf"
    text = load_pdf(pdf_path)

    print(f"loaded text length: {len(text)} characters from pdf")
    print(f"first 500 characters: {text[:500]}")
    print("\n" + "="*50 + "\n")


# chunk the text into smaller pieces
    chunks = chunk_text(text,chunk_size=500, overlap=50)
    print(f"Created {len(chunks)} chunks")
    print(f"\nFirst chunk:\n{chunks[0]}")
    print(f"\nSecond chunk:\n{chunks[1]}")
    print(f"\nThird chunk:\n{chunks[2]}")






