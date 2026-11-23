"""Copy the registration evidence into registration_evidence/ so an archive (Zenodo) holds it outside GitHub.

A release's source zip has no git history, and GitHub's push log lives only
on GitHub. This writes:
  github_push_log.json - GitHub's activity log for the repository, as
                         returned by its API (server-side push times)
  git_log.txt          - every commit with its hash, tree, parents, author
                         and committer times, and the files it changed
  repo.bundle          - the full git history up to this commit
                         (`git clone repo.bundle` rebuilds it, and every
                         commit hash in the README and the push log can be
                         checked against it, without GitHub)
  SHA256SUMS           - checksums of the three files
Then run check_registrations.py as usual. Run in a full clone.
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import hashlib
import json
import os
import subprocess

from check_registrations import activity

OUT = "registration_evidence"

if __name__ == "__main__":
    out = write_path(OUT)                  # the whole folder: rerun/registration_evidence unless --force
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/github_push_log.json", "w") as f:
        json.dump(activity(), f, indent=1)
    log = subprocess.run(["git", "log", "--format=fuller", "--format=commit %H%ntree %T%nparents %P%n"
                          "author %an %aI%ncommitter %cn %cI%n%n    %s%n", "--name-status", "HEAD"],
                         capture_output=True, text=True, check=True).stdout
    open(f"{out}/git_log.txt", "w").write(log)
    subprocess.run(["git", "bundle", "create", f"{out}/repo.bundle", "HEAD", "main"], check=True,
                   capture_output=True)
    with open(f"{out}/SHA256SUMS", "w") as f:
        for name in ["github_push_log.json", "git_log.txt", "repo.bundle"]:
            f.write(f"{hashlib.sha256(open(f'{out}/{name}', 'rb').read()).hexdigest()}  {name}\n")
    print(open(f"{out}/SHA256SUMS").read())
