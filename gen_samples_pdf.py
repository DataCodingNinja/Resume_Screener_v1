import os
import random
import csv
import argparse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

SKILLS = ["Python","SQL","Excel","Tableau","Power BI","Pandas","NumPy","scikit-learn",
          "Matplotlib","Seaborn","Statistics","Regression","ETL","AWS","BigQuery","Spark"]
TITLES = ["Data Analyst","Business Analyst","Junior Data Analyst","Research Analyst",
          "Data Scientist","Analytics Engineer","Reporting Analyst"]
EDU = ["Bachelors in Computer Science","Bachelors in Statistics","B.Sc. Mathematics",
       "MSc Data Science","Masters in Business Analytics","Bachelors in Economics"]
COMPANIES = ["Acme Corp","BrightData","NextGen Analytics","InfoWorks","DataHive","SparkSoft"]

def random_years_exp():
    return random.choice([0,1,2,3,4,5,6,7,8])

def make_resume_text(i, role="Data Analyst"):
    name = f"Candidate {i}"
    title = random.choice(TITLES) if random.random() > 0.2 else role
    years = random_years_exp()
    education = random.choice(EDU)
    skills_count = random.choices([1,2,3,4,5,6,7,8], weights=[5,10,20,25,20,12,5,3])[0]
    skills = random.sample(SKILLS, skills_count)
    projects = []
    proj_n = random.choice([0,1,1,2])
    for p in range(proj_n):
        proj_skills = random.sample(skills, k=min(len(skills), random.randint(1,3)))
        projects.append(f"Project {p+1}: Built dashboard using {', '.join(proj_skills)} for business insights.")
    exp_lines = []
    jobs = random.choice([1,1,2,2,3])
    # create simple date ranges
    start_year = 2025 - (years + random.randint(0,2))
    for j in range(jobs):
        dur = max(1, int(years/jobs)) if years>0 else 1
        end_year = start_year + dur
        exp_lines.append(f"{start_year} - {end_year}: {title} at {random.choice(COMPANIES)} - worked on {', '.join(random.sample(skills, min(2,len(skills))))}.")
        start_year = end_year + 1
    lines = []
    lines.append(name)
    lines.append(title)
    lines.append(education)
    lines.append(f"{years} years experience")
    lines.append("Skills: " + ", ".join(skills))
    lines.extend(exp_lines)
    lines.extend(projects)
    lines.append("Certifications: " + (random.choice(["None","Google Data Analytics","AWS Certified Data Analytics","Tableau Desktop Specialist"])))
    return "\n".join(lines), skills, years, title

def write_pdf(text, path):
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    margin = 50
    y = height - margin
    line_height = 12
    for line in text.split("\n"):
        # wrap long lines
        if len(line) <= 90:
            c.drawString(margin, y, line)
            y -= line_height
        else:
            # naive wrap
            parts = [line[i:i+90] for i in range(0, len(line), 90)]
            for p in parts:
                c.drawString(margin, y, p)
                y -= line_height
        if y < margin + 40:
            c.showPage()
            y = height - margin
    c.save()

def generate(folder, n):
    os.makedirs(folder, exist_ok=True)
    gt_rows = []
    for i in range(1, n+1):
        text, skills, years, title = make_resume_text(i)
        fname = f"resume_{i}.pdf"
        path = os.path.join(folder, fname)
        write_pdf(text, path)
        # label as fit if has >=3 key data analyst skills and >=1 year exp or title contains Data Analyst
        key_skills = {"python","sql","excel","pandas","numpy","tableau","statistics"}
        have = len(key_skills.intersection(set([s.lower() for s in skills])))
        label = 1 if (have >= 3 and years >= 1) or ("data analyst" in title.lower()) else 0
        gt_rows.append([fname, label, "Data Analyst"])
    gt_path = os.path.join(folder, "ground_truth.csv")
    with open(gt_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["resume_id","label","role"])
        writer.writerows(gt_rows)
    print(f"Generated {n} PDF resumes in {folder} and ground_truth.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50, help="number of sample resumes to create")
    p.add_argument("--out", type=str, default="sample_resumes_pdf", help="output folder")
    args = p.parse_args()
    generate(args.out, args.n)
