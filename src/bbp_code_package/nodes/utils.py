from PyPDF2 import PdfFileMerger


def merge_pdfs(file_list: list, writing_path: str):
    """
    Merge the listed pdfs in one unique file
    :param file_list: list of file paths
    :return: One unique pdf
    """
    merger = PdfFileMerger()

    # Iterate over the list of the file paths
    for pdf_file in file_list:
        # Append PDF files
        merger.append(pdf_file)

    # Write out the merged PDF file
    merger.write(writing_path)
    merger.close()
