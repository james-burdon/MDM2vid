import json
import time
import os
import sys
import pandas as pd
from linkedin_api import Linkedin

# Set up LinkedIn credentials
LINKEDIN_USERNAME = os.environ.get("LINKEDIN_USERNAME")
LINKEDIN_PASSWORD = os.environ.get("LINKEDIN_PASSWORD")

OUTPUT_CSV = "linkedin_jobs_output.csv"


def scrape_linkedin_jobs(keywords="Data Scientist", location="United States", limit=10):
    if not (LINKEDIN_USERNAME and LINKEDIN_PASSWORD):
        print("LinkedIn credentials not found. Exiting...")
        sys.exit()

    api = Linkedin(LINKEDIN_USERNAME, LINKEDIN_PASSWORD)
    jobs = api.search_jobs(keywords=keywords, location_name=location, limit=limit)

    job_list = []

    for job in jobs:
        job_id = job["entityUrn"].split(":")[-1]
        details = api.get_job(job_id)

        company = (
            details.get("companyResolutionResult", {}).get("name") or
            details.get("companyDetails", {})
            .get("com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany", {})
            .get("companyResolutionResult", {})
            .get("name") or
            "Unknown"
        )

        description_data = details.get("description", {})
        description = description_data.get("text", "")
        description = " ".join(description.split())

        job_entry = {
            "Job Title": details.get("title", "Unknown"),
            "Company Name": company,
            "Location": details.get("formattedLocation", "Unknown"),
            "Job Description": description if description else "No description available",
            "Salary Estimate": details.get("salaryInsight", {}).get("salary", "Not Provided"),
            "Industry": details.get("industry", "Unknown"),
            "Job Type": details.get("employmentStatus", "Unknown"),
            "Experience Level": details.get("experienceLevel", "Unknown"),
        }
        job_list.append(job_entry)

        time.sleep(2)  # Avoid hitting LinkedIn's rate limits

    # Save to CSV
    df = pd.DataFrame(job_list)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Job listings saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    scrape_linkedin_jobs()



