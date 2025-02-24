import json
import time
import os
import sys
import pandas as pd
import random
import re
from concurrent.futures import ThreadPoolExecutor
from linkedin_api import Linkedin

# Set up LinkedIn credentials
LINKEDIN_USERNAME = os.environ.get("LINKEDIN_USERNAME")
LINKEDIN_PASSWORD = os.environ.get("LINKEDIN_PASSWORD")

OUTPUT_CSV = "linkedin_jobs_output.csv"

def extract_from_description(description, field):
    """Extract salary, experience, and job type from job descriptions using regex & keyword matching"""
    if not description:
        return "Unknown"

    description = description.lower()[:500]  # Limit processing to 500 characters for speed
    
    # Extract salary from text (match formats like "$70K", "$120,000", etc.)
    if field == "salary":
        match = re.search(r"\$\d{2,3}[kK]|\$\d{1,3},\d{3}", description)
        return match.group(0) if match else "Not Provided"

    # Extract experience level from text
    if field == "experience":
        levels = ["entry", "junior", "mid", "senior", "lead", "executive"]
        for level in levels:
            if level in description:
                return level.capitalize()
        return "Unknown"

    # Extract job type from text
    if field == "job_type":
        job_types = ["full-time", "part-time", "contract", "intern", "remote", "temporary"]
        for jt in job_types:
            if jt in description:
                return jt.capitalize()
        return "Unknown"

    return "Unknown"

def fetch_job_details(api, job, retries=1):  # Reduced retries to 1 for faster execution
    """Fetch job details with optimized fallback extraction logic"""
    job_id = job["entityUrn"].split(":")[-1]
    
    for attempt in range(retries):
        try:
            details = api.get_job(job_id)

            # Extract company name
            company = (
                details.get("companyResolutionResult", {}).get("name") or
                details.get("companyDetails", {}).get("name") or
                "Unknown"
            )

            # Extract job description
            description_data = details.get("description", {})
            description = description_data.get("text", "").strip()
            description = " ".join(description.split())[:500]  # Limit processing

            # Extract salary more accurately, fallback to description scan
            salary_data = details.get("salaryInsight", {})
            salary = (
                salary_data.get("salary") or
                salary_data.get("range", {}).get("min") or
                salary_data.get("range", {}).get("max") or
                salary_data.get("range", {}).get("median") or
                extract_from_description(description, "salary")  # Fallback extraction from description
            )

            # Extract job type, fallback to description scan
            job_type = (
                details.get("employmentType") or
                details.get("employmentStatus") or
                details.get("jobPosting", {}).get("employmentType") or
                extract_from_description(description, "job_type")  # Fallback extraction from description
            )

            # Extract experience level, fallback to description scan
            experience_level = (
                details.get("experienceLevel") or
                details.get("jobPosting", {}).get("experienceLevel") or
                details.get("jobPosting", {}).get("requiredExperienceLevel") or
                extract_from_description(description, "experience")  # Fallback extraction from description
            )

            return {
                "Job Title": details.get("title", "Unknown"),
                "Company Name": company,
                "Location": details.get("formattedLocation", "Unknown"),
                "Job Description": description if description else "No description available",
                "Salary Estimate": salary,
                "Job Type": job_type,
                "Experience Level": experience_level,
            }
        except Exception as e:
            print(f"Error fetching job {job_id}, attempt {attempt+1}/{retries}: {e}")
            time.sleep(random.uniform(0.5, 2))  # Reduce retry wait time to 0.5-2 seconds

    return None  # Return None if all attempts fail

def scrape_linkedin_jobs(job_titles=["Data Scientist"], location="United States", limit=15):  # Lowered limit for speed
    """Scrape LinkedIn jobs using faster multithreading"""
    if not (LINKEDIN_USERNAME and LINKEDIN_PASSWORD):
        print("LinkedIn credentials not found. Exiting...")
        sys.exit()

    api = Linkedin(LINKEDIN_USERNAME, LINKEDIN_PASSWORD)
    job_list = []

    for title in job_titles:
        print(f"Searching for {title} jobs in {location}...")
        try:
            jobs = api.search_jobs(keywords=title, location_name=location, limit=limit)
        except Exception as e:
            print(f"Failed to fetch job listings for {title}: {e}")
            continue

        with ThreadPoolExecutor(max_workers=15) as executor:  # Increased workers to 15 for max speed
            results = list(executor.map(lambda job: fetch_job_details(api, job), jobs))

        job_list.extend(filter(None, results))  # Remove None values from failed jobs

    # Save to CSV (append mode)
    df = pd.DataFrame(job_list)
    df.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
    print(f"Job listings saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    job_roles = ["Data Scientist", "Software Engineer", "Financial Analyst", "Product Manager"]
    scrape_linkedin_jobs(job_titles=job_roles, location="United States", limit=15)  # Lower limit for speed
