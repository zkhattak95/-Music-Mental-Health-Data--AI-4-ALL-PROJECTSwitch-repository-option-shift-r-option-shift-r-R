# Start Here

## Fastest path to a working demo

1. Copy every file and folder in this starter into the cloned GitHub repository.
2. Do **not** copy the 1,000,000-row burnout CSV into GitHub.
3. Open the repository in VS Code.
4. Run:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

5. Test all five pages.
6. Commit your changes on a feature branch:

```bash
git checkout -b feature/mind-and-melody-app
git add .
git commit -m "Build Mind and Melody Streamlit web app"
git push -u origin feature/mind-and-melody-app
```

## Before the Friday deadline

- Confirm every team member's name is spelled correctly.
- Replace the working title if the group chooses another name.
- Test the burnout form with low, medium, and high input patterns.
- Review the responsible-use wording as a team.
- Take screenshots for the final presentation.
- Deploy only after the local app works.
