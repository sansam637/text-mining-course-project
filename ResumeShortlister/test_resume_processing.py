"""Test script to verify resume processing functionality"""
from pathlib import Path
from app import create_app

# Create app context
app = create_app('development')

with app.app_context():
    from app.services.resume_parser import ResumeParserFactory
    from app.services.file_service import FileService
    from app.services.matching_service import ResumeMatchingService
    from app.models.job_description import JobDescription
    
    print("Testing Resume Processing...")
    print("=" * 50)
    
    # Test file service
    file_service = FileService()
    resume_files = file_service.get_resume_files()
    print(f"✓ Found {len(resume_files)} resume files")
    
    # Test parsing one resume
    if resume_files:
        resume_path = resume_files[0]
        print(f"\nTesting resume: {resume_path.name}")
        try:
            parser = ResumeParserFactory.get_parser(resume_path.suffix)
            candidate = parser.parse(resume_path)
            print(f"✓ Successfully parsed: {candidate.display_name}")
            print(f"  Email: {candidate.email}")
            print(f"  Skills: {candidate.skills[:5] if candidate.skills else 'None'}")
        except Exception as e:
            print(f"✗ Error parsing resume: {e}")
            import traceback
            traceback.print_exc()
    
    # Test job description parsing
    print(f"\n" + "=" * 50)
    print("Testing Job Description...")
    job_file = Path('data/job_descriptions/amazon.txt')
    if job_file.exists():
        try:
            with open(job_file, 'r', encoding='utf-8') as f:
                content = f.read()
            job_description = JobDescription(
                title=job_file.stem,
                description=content,
                file_path=job_file
            )
            print(f"✓ Successfully loaded job description: {job_description.display_name}")
        except Exception as e:
            print(f"✗ Error loading job description: {e}")
            import traceback
            traceback.print_exc()
    
    # Test full matching process
    print(f"\n" + "=" * 50)
    print("Testing Full Matching Process...")
    try:
        candidates = []
        for resume_file in resume_files[:3]:  # Test first 3 resumes
            try:
                parser = ResumeParserFactory.get_parser(resume_file.suffix)
                candidate = parser.parse(resume_file)
                candidates.append(candidate)
            except Exception as e:
                print(f"  Warning: Failed to parse {resume_file.name}: {e}")
        
        if candidates and job_description:
            matching_service = ResumeMatchingService(
                top_candidates_count=10,
                similarity_threshold=0.0  # Set to 0 to see all candidates
            )
            ranked_candidates = matching_service.match_candidates(job_description, candidates)
            print(f"✓ Successfully matched {len(ranked_candidates)} candidates")
            print(f"  Total candidates parsed: {len(candidates)}")
            
            if ranked_candidates:
                print("\nTop candidates:")
                for i, candidate in enumerate(ranked_candidates[:5], 1):
                    print(f"  {i}. {candidate.display_name} - Score: {candidate.score:.3f}")
            else:
                print("\n  No candidates returned (all below threshold or parsing failed)")
        else:
            print("✗ No candidates or job description available for matching")
            
    except Exception as e:
        print(f"✗ Error in matching process: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "=" * 50)
    print("Test completed!")
