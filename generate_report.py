from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime

helmet = int(input("Helmet violations: "))
vest = int(input("Vest violations: "))
mask = int(input("Mask violations: "))

file = f"reports/PPE_Report_{datetime.date.today()}.pdf"

c = canvas.Canvas(file, pagesize=letter)

c.drawString(200,750,"PPE Safety Report")
c.drawString(50,700,f"Date: {datetime.date.today()}")

c.drawString(50,650,f"No Helmet Violations: {helmet}")
c.drawString(50,620,f"No Vest Violations: {vest}")
c.drawString(50,590,f"No Mask Violations: {mask}")

c.save()

print("Report Generated:", file)