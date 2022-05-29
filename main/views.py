from django.shortcuts import render

def main_page(request):
    """Main page"""

    return render(request, "main.html")