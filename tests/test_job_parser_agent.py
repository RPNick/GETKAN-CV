import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.job_parser_agent import (
    JobParserState,
    _load_dotenv,
    extract_facts,
    fetch_or_load_listing,
    handoff_to_tailor,
    normalize_packet,
    validate_packet,
)
from src.agents.job_hunt_advisor import generate_job_hunt_recommendations
from src.agents.resume_tailor_agent import _score_item, build_allowlist, build_tailored_payload
from src.main import calculate_compatibility_score, run


class JobParserAgentTests(unittest.TestCase):
    def test_calculate_compatibility_score_partial_long_requirement(self):
        score = calculate_compatibility_score(
            {
                "job": {
                    "title": "",
                    "domain": "",
                    "must_have": ["Strong programming skills with languages like Rust, Go, or Python"],
                    "nice_to_have": [],
                }
            }
        )
        self.assertGreaterEqual(score, 3)

    def test_calculate_compatibility_score_returns_valid_range(self):
        score = calculate_compatibility_score(
            {
                "job": {
                    "title": "Senior Backend Engineer",
                    "domain": "Cloud Infrastructure",
                    "must_have": ["Python", "Docker", "Kubernetes"],
                    "nice_to_have": ["Terraform", "Go"],
                }
            }
        )
        self.assertGreaterEqual(score, 1)
        self.assertLessEqual(score, 10)

    def test_job_hunt_advisor_without_packets_writes_general_advice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "output"
            modules_dir = Path(tmpdir) / "modules"
            modules_dir.mkdir(parents=True, exist_ok=True)
            (modules_dir / "summary.tex").write_text("Senior engineer profile.", encoding="utf-8")
            (modules_dir / "experience.tex").write_text("Built APIs and services.", encoding="utf-8")
            (modules_dir / "personalprojects.tex").write_text("Projects.", encoding="utf-8")
            (modules_dir / "aboutme.tex").write_text("About me.", encoding="utf-8")

            result = generate_job_hunt_recommendations(output_root=output_root, resume_modules_dir=modules_dir)

            self.assertEqual(result["packet_count"], 0)
            recommendations_text = Path(result["recommendations_path"]).read_text(encoding="utf-8")
            self.assertIn("General Recommendations (No Job Packets Provided)", recommendations_text)
            self.assertIn("Suggested Technologies And Languages To Work On", recommendations_text)
            self.assertIn("Kubernetes", recommendations_text)

    def test_job_hunt_advisor_writes_recommendations_from_saved_packets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "output"
            packet_dir = output_root / "sample-role"
            packet_dir.mkdir(parents=True, exist_ok=True)
            (packet_dir / "job_packet.json").write_text(
                json.dumps(
                    {
                        "job": {
                            "title": "Staff Backend Engineer",
                            "domain": "developer tools",
                            "description": "Design backend APIs and cloud services",
                            "must_have": ["Python", "Kubernetes"],
                            "nice_to_have": ["Terraform", "GraphQL"],
                            "responsibilities": ["Own reliability and CI/CD pipelines"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            modules_dir = Path(tmpdir) / "modules"
            modules_dir.mkdir(parents=True, exist_ok=True)
            (modules_dir / "summary.tex").write_text("Experienced engineer with Python and API delivery.", encoding="utf-8")
            (modules_dir / "experience.tex").write_text("Built backend services and mentored engineers.", encoding="utf-8")
            (modules_dir / "personalprojects.tex").write_text("Project work in automation.", encoding="utf-8")
            (modules_dir / "aboutme.tex").write_text("Hands-on builder.", encoding="utf-8")
            (modules_dir / "skills.json").write_text(
                json.dumps(
                    {
                        "programming_languages": ["Python", "TypeScript"],
                        "devops_and_delivery": ["GitHub Actions", "Docker"],
                    }
                ),
                encoding="utf-8",
            )

            result = generate_job_hunt_recommendations(output_root=output_root, resume_modules_dir=modules_dir)

            self.assertEqual(result["packet_count"], 1)
            self.assertTrue(Path(result["recommendations_path"]).exists())
            recommendations_text = Path(result["recommendations_path"]).read_text(encoding="utf-8")
            self.assertIn("Highest-Impact Skills To Build Or Show", recommendations_text)
            self.assertIn("Kubernetes", recommendations_text)

    def test_category_boosts_raise_testing_bullet_score(self):
        item = "Established testing framework with Jest and Vue Test Utils for unit and integration test coverage"
        baseline = _score_item(item, [])
        boosted = _score_item(
            item,
            [],
            category_skills={"testing_and_quality": ["jest", "vue test utils", "unit testing", "integration testing"]},
            category_boosts={"testing_and_quality": 3},
        )
        self.assertGreater(boosted, baseline)

    def test_build_allowlist_uses_skills_file(self):
        state = {
            "job_packet": {},
            "source_modules": {},
            "allowlist": [],
            "prompts": {},
            "model_output": {},
            "violations": [],
            "compile_log": "",
        }
        build_allowlist(state)

        allowlist = [item.lower() for item in state["allowlist"]]
        self.assertIn("javascript", allowlist)
        self.assertIn("laravel", allowlist)
        self.assertIn("github actions", allowlist)
        self.assertIn("jest", allowlist)

    def test_parser_pipeline_extracts_and_normalizes_listing(self):
        listing_text = """
        Senior Software Engineer
        ExampleCo
        Austin, TX
        Full-time

        We are looking for a Senior Software Engineer with experience in Python, Django, and AWS.
        Responsibilities include building APIs, mentoring engineers, and improving platform reliability.
        Must have: Python, Django, AWS
        Nice to have: Kubernetes, Terraform
        """

        state: JobParserState = {
            "source": {"job_url": "https://example.com/jobs/123", "listing_text": listing_text},
            "raw_listing_text": listing_text,
            "extracted_facts": {},
            "normalized_packet": {},
            "confidence": 0.0,
        }

        fetch_or_load_listing(state)

        with patch(
            "src.agents.job_parser_agent._parse_job_with_openrouter",
            return_value={
                "title": "Senior Software Engineer",
                "company": "ExampleCo",
                "location": "Austin, TX",
                "employment_type": "Full-time",
                "description": "Build APIs and tooling",
                "must_have": ["Python", "Django", "AWS"],
                "nice_to_have": ["Kubernetes", "Terraform"],
                "responsibilities": ["Build APIs"],
                "domain": "saas",
            },
        ):
            extract_facts(state)
            normalize_packet(state)
            validate_packet(state)

        self.assertIn("title", state["normalized_packet"]["job"])
        self.assertEqual(state["normalized_packet"]["job"]["company"], "ExampleCo")
        self.assertIn("Python", state["normalized_packet"]["job"]["must_have"])
        self.assertIn("AWS", state["normalized_packet"]["job"]["must_have"])
        self.assertGreaterEqual(state["confidence"], 0.5)

    def test_load_dotenv_reads_repo_environment_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")
            os.environ.pop("OPENROUTER_API_KEY", None)
            _load_dotenv(env_path)
            self.assertEqual(os.environ["OPENROUTER_API_KEY"], "test-key")
            os.environ.pop("OPENROUTER_API_KEY", None)

    def test_extract_facts_prefers_openrouter_payload_when_available(self):
        state: JobParserState = {
            "source": {"job_url": "https://example.com/jobs/789"},
            "raw_listing_text": "Principal Platform Engineer\nNorthwind\nRemote\n",
            "extracted_facts": {},
            "normalized_packet": {},
            "confidence": 0.0,
        }

        with patch(
            "src.agents.job_parser_agent._parse_job_with_openrouter",
            return_value={
                "title": "Principal Platform Engineer",
                "company": "Northwind",
                "location": "Remote",
                "employment_type": "Full-time",
                "description": "Build platform tooling",
                "must_have": ["Python", "Kubernetes"],
                "nice_to_have": ["Terraform"],
                "responsibilities": ["Lead platform work"],
                "domain": "saas",
            },
        ):
            extract_facts(state)

        self.assertEqual(state["extracted_facts"]["company"], "Northwind")
        self.assertIn("Python", state["extracted_facts"]["must_have"])

    def test_handoff_writes_job_packet_to_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state: JobParserState = {
                "source": {"job_url": "https://example.com/jobs/456"},
                "raw_listing_text": "Staff Product Engineer\nAcme\nRemote\n",
                "extracted_facts": {
                    "title": "Staff Product Engineer",
                    "company": "Acme",
                    "location": "Remote",
                    "description": "Build products",
                    "must_have": ["TypeScript"],
                    "nice_to_have": ["React"],
                    "responsibilities": ["Ship features"],
                    "domain": "SaaS",
                },
                "normalized_packet": {},
                "confidence": 0.79,
            }

            result = handoff_to_tailor(state, output_dir=tmpdir)
            output_path = Path(tmpdir) / "job_packet.json"

            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text())
            self.assertEqual(payload["job"]["title"], "Staff Product Engineer")
            self.assertEqual(result["output_path"], str(output_path))

    def test_run_uses_current_working_directory_for_default_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            listing_path = Path(tmpdir) / "listing.txt"
            listing_path.write_text("Example listing", encoding="utf-8")

            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with patch("src.main.fetch_or_load_listing"), patch("src.main.extract_facts"), patch("src.main.normalize_packet"), patch("src.main.validate_packet"), patch(
                    "src.main.handoff_to_tailor", return_value={"output_path": str(Path(tmpdir) / "job_packet.json")}
                ), patch("src.main.build_tailored_payload", return_value={"ok": True}):
                    exit_code = run("demo-job", str(listing_path), None, None, None)
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(exit_code, 0)
            output_dir = Path(tmpdir) / "output" / "demo-job"
            self.assertTrue(output_dir.exists())
            self.assertTrue((output_dir / "tailored_resume.json").exists())

    def test_run_writes_source_history_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            listing_path = Path(tmpdir) / "listing.txt"
            listing_path.write_text("Example listing", encoding="utf-8")

            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with patch("src.main.fetch_or_load_listing"), patch("src.main.extract_facts"), patch("src.main.normalize_packet"), patch("src.main.validate_packet"), patch(
                    "src.main.handoff_to_tailor", return_value={"output_path": str(Path(tmpdir) / "job_packet.json")}
                ), patch("src.main.build_tailored_payload", return_value={"ok": True, "compile": {"pdf_path": ""}}):
                    exit_code = run("demo-job", str(listing_path), "https://example.com/jobs/1", None, "gpt-4o-mini")
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(exit_code, 0)
            log_path = Path(tmpdir) / "log" / "source_history.jsonl"
            self.assertTrue(log_path.exists())
            lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreaterEqual(len(lines), 1)
            payload = json.loads(lines[-1])
            self.assertEqual(payload.get("job_name"), "demo-job")
            self.assertEqual(payload.get("url"), "https://example.com/jobs/1")
            self.assertTrue(payload.get("file", "").endswith("listing.txt"))
            self.assertEqual(payload.get("model_name"), "gpt-4o-mini")
            self.assertIsInstance(payload.get("compatibility_score"), int)
            self.assertGreaterEqual(payload.get("compatibility_score"), 1)
            self.assertLessEqual(payload.get("compatibility_score"), 10)

    def test_run_recompile_mode_skips_parsing_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output" / "demo-job"
            output_dir.mkdir(parents=True, exist_ok=True)

            with patch("src.main.fetch_or_load_listing") as fetch_mock, patch("src.main.extract_facts") as extract_mock, patch(
                "src.main.normalize_packet"
            ) as normalize_mock, patch("src.main.validate_packet") as validate_mock, patch("src.main.handoff_to_tailor") as handoff_mock, patch(
                "src.main.recompile_existing_output",
                return_value={"summary": str(output_dir / "tailored_resume.json"), "compile": {"pdf_path": str(output_dir / "demo-job.pdf"), "page_count": 1}},
            ) as recompile_mock:
                exit_code = run("demo-job", None, None, str(output_dir), None, recompile_existing=True)

            self.assertEqual(exit_code, 0)
            recompile_mock.assert_called_once()
            fetch_mock.assert_not_called()
            extract_mock.assert_not_called()
            normalize_mock.assert_not_called()
            validate_mock.assert_not_called()
            handoff_mock.assert_not_called()

    def test_run_build_basic_mode_skips_parsing_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.main.fetch_or_load_listing") as fetch_mock, patch("src.main.extract_facts") as extract_mock, patch(
                "src.main.normalize_packet"
            ) as normalize_mock, patch("src.main.validate_packet") as validate_mock, patch("src.main.handoff_to_tailor") as handoff_mock, patch(
                "src.main.build_basic_resume",
                return_value={"output_dir": str(Path(tmpdir) / "output" / "general"), "pdf": str(Path(tmpdir) / "output" / "general" / "resume.pdf"), "compile_log": ""},
            ) as basic_mock:
                exit_code = run(None, None, None, str(Path(tmpdir) / "output" / "general"), None, build_basic=True)

            self.assertEqual(exit_code, 0)
            basic_mock.assert_called_once()
            fetch_mock.assert_not_called()
            extract_mock.assert_not_called()
            normalize_mock.assert_not_called()
            validate_mock.assert_not_called()
            handoff_mock.assert_not_called()

    def test_run_job_hunt_advice_mode_skips_parsing_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.main.fetch_or_load_listing") as fetch_mock, patch("src.main.extract_facts") as extract_mock, patch(
                "src.main.normalize_packet"
            ) as normalize_mock, patch("src.main.validate_packet") as validate_mock, patch("src.main.handoff_to_tailor") as handoff_mock, patch(
                "src.main.generate_job_hunt_recommendations",
                return_value={
                    "recommendations_path": str(Path(tmpdir) / "output" / "job_hunt_recommendations.md"),
                    "packet_count": 2,
                    "missing_skills": [],
                    "matched_skills": [],
                },
            ) as advice_mock:
                exit_code = run(None, None, None, str(Path(tmpdir) / "output"), None, job_hunt_advice=True)

            self.assertEqual(exit_code, 0)
            advice_mock.assert_called_once()
            fetch_mock.assert_not_called()
            extract_mock.assert_not_called()
            normalize_mock.assert_not_called()
            validate_mock.assert_not_called()
            handoff_mock.assert_not_called()

    def test_run_job_hunt_advice_mode_passes_explicit_packet_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            packet_path = str(Path(tmpdir) / "job_packet.json")
            with patch("src.main.generate_job_hunt_recommendations", return_value={"recommendations_path": "x", "packet_count": 0}) as advice_mock:
                exit_code = run(None, None, None, str(Path(tmpdir) / "output"), None, job_hunt_advice=True, job_packet_files=[packet_path])

            self.assertEqual(exit_code, 0)
            advice_mock.assert_called_once()
            _, kwargs = advice_mock.call_args
            self.assertEqual(kwargs.get("job_packet_files"), [packet_path])

    def test_run_batch_urls_from_file_auto_generates_job_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                list_file = Path(tmpdir) / "urls.txt"
                list_file.write_text("https://example.com/job-a\nhttps://example.com/job-b\n", encoding="utf-8")

                def normalize_side_effect(state):
                    url = state.get("source", {}).get("job_url", "")
                    if "job-a" in url:
                        state["normalized_packet"] = {
                            "job": {
                                "company": "Acme",
                                "title": "Senior Backend Engineer",
                                "must_have": ["Python"],
                                "nice_to_have": ["Docker"],
                                "domain": "SaaS",
                            }
                        }
                    else:
                        state["normalized_packet"] = {
                            "job": {
                                "company": "Acme",
                                "title": "Senior Backend Engineer",
                                "must_have": ["Python"],
                                "nice_to_have": ["Kubernetes"],
                                "domain": "SaaS",
                            }
                        }

                def handoff_side_effect(state, output_dir):
                    return {"output_path": str(Path(output_dir) / "job_packet.json")}

                def payload_side_effect(job_packet, job_name, output_dir, model_name=None):
                    return {"compile": {"pdf_path": str(Path(output_dir) / f"{job_name}.pdf")}}

                with patch("src.main.fetch_or_load_listing"), patch("src.main.extract_facts"), patch("src.main.normalize_packet", side_effect=normalize_side_effect), patch(
                    "src.main.validate_packet"
                ), patch("src.main.handoff_to_tailor", side_effect=handoff_side_effect) as handoff_mock, patch(
                    "src.main.build_tailored_payload", side_effect=payload_side_effect
                ) as payload_mock:
                    exit_code = run(
                        None,
                        None,
                        None,
                        None,
                        None,
                        url_list_file=str(list_file),
                    )

                self.assertEqual(exit_code, 0)
                self.assertEqual(handoff_mock.call_count, 2)
                self.assertEqual(payload_mock.call_count, 2)

                log_path = Path(tmpdir) / "log" / "source_history.jsonl"
                self.assertTrue(log_path.exists())
                entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertGreaterEqual(len(entries), 2)
                last_two = entries[-2:]
                names = [entry["job_name"] for entry in last_two]
                self.assertEqual(names[0], "acme-senior-backend-engineer")
                self.assertEqual(names[1], "acme-senior-backend-engineer-2")
            finally:
                os.chdir(previous_cwd)

    def test_build_tailored_payload_writes_summary_modules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = build_tailored_payload(
                {
                    "job": {
                        "title": "Senior Software Engineer",
                        "company": "GitHub",
                        "location": "Remote",
                        "employment_type": "Full-time",
                        "description": "Build billing and platform systems",
                        "must_have": ["Python", "AWS", "Kubernetes"],
                        "nice_to_have": ["TypeScript"],
                        "responsibilities": ["Lead backend architecture"],
                        "domain": "developer tools",
                    },
                    "metadata": {},
                },
                job_name="demo-job",
                output_dir=tmpdir,
            )

            self.assertIn("summary.tex", payload["model_output"]["tailored_modules"])
            self.assertIn("Senior Software Engineer", payload["model_output"]["tailored_modules"]["summary.tex"])
            self.assertNotIn("relevance_notes.txt", payload["model_output"]["tailored_modules"])
            self.assertNotIn("experience_highlights.tex", payload["model_output"]["tailored_modules"])
            self.assertIn("experience.tex", payload["model_output"]["tailored_modules"])
            self.assertIn("personalprojects.tex", payload["model_output"]["tailored_modules"])
            self.assertIn("aboutme.tex", payload["model_output"]["tailored_modules"])
            self.assertIn("Bachelor of Arts in Computer Science", payload["model_output"]["tailored_modules"]["aboutme.tex"])
            self.assertIn("Bachelor of Arts in Economics", payload["model_output"]["tailored_modules"]["aboutme.tex"])
            self.assertTrue(Path(tmpdir, "resume", "modules", "summary.tex").exists())
            self.assertTrue(Path(tmpdir, "resume", "modules", "experience.tex").exists())
            self.assertTrue(Path(tmpdir, "resume", "modules", "personalprojects.tex").exists())
            self.assertTrue(Path(tmpdir, "resume", "modules", "aboutme.tex").exists())
            self.assertTrue(Path(tmpdir, "demo-job.pdf").exists())
            self.assertTrue(Path(tmpdir, "tailored_resume.json").exists())

    def test_extract_facts_falls_back_when_openrouter_returns_unknown_values(self):
        state: JobParserState = {
            "source": {"job_url": "https://example.com/jobs/999"},
            "raw_listing_text": "Title: Senior Software Engineer\nCompany: GitHub\nDescription: Build reliable billing systems\nPython AWS",
            "extracted_facts": {},
            "normalized_packet": {},
            "confidence": 0.0,
        }

        with patch(
            "src.agents.job_parser_agent._parse_job_with_openrouter",
            return_value={
                "title": "unknown",
                "company": "unknown",
                "location": "unknown",
                "employment_type": "unknown",
                "description": "unknown",
                "must_have": [],
                "nice_to_have": [],
                "responsibilities": [],
                "domain": "unknown",
            },
        ):
            extract_facts(state)

        self.assertEqual(state["extracted_facts"]["title"], "Senior Software Engineer")
        self.assertEqual(state["extracted_facts"]["company"], "GitHub")
        self.assertIn("Python", state["extracted_facts"]["must_have"])
        self.assertIn("AWS", state["extracted_facts"]["must_have"])

    def test_extract_facts_uses_fallback_employment_type(self):
        state: JobParserState = {
            "source": {"job_url": "https://example.com/jobs/1000"},
            "raw_listing_text": "Title: Senior Software Engineer\nEmployment Type: Full Time\nCompany: GitHub\n",
            "extracted_facts": {},
            "normalized_packet": {},
            "confidence": 0.0,
        }

        with patch(
            "src.agents.job_parser_agent._parse_job_with_openrouter",
            return_value={
                "title": "unknown",
                "company": "unknown",
                "location": "unknown",
                "employment_type": "unknown",
                "description": "unknown",
                "must_have": [],
                "nice_to_have": [],
                "responsibilities": [],
                "domain": "unknown",
            },
        ):
            extract_facts(state)

        self.assertEqual(state["extracted_facts"]["employment_type"], "Full Time")


if __name__ == "__main__":
    unittest.main()
