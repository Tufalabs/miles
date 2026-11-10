# Tufa Fork

Applies repository-wide, to changes and to their review. Tufa only: upstream
(`radixark/miles`) has no such file, so it never conflicts at a sync.

This repository is Tufa's fork of Miles. `main` mirrors upstream and is never
committed to; our changes merge into `tufa/main`. rave pins a commit of
`tufa/main` and every training run on Thur uses the pinned code, so a change
here reaches every run at the next pin move.

## Changes

- Change Miles only when rave cannot do it from outside, and keep the change
  small enough to send upstream.
- Put new behaviour behind a new flag whose default keeps today's behaviour.
- Prefer a new file or a flag-gated branch to edits spread through an upstream
  file. Each edited upstream line is a possible conflict at the next sync.
- A cherry-pick of an upstream pull request names it in the title
  (`radixark/miles#<n>`) and stays as close to the upstream commit as possible.

## Review

For whoever reviews a pull request: the Claude review that runs when a pull
request is marked ready, `@claude review` or `@codex review` in a comment, and
people. Report these before anything from the other rule files, and not as
Nits:

- A change in behaviour for a run that did not pass a new flag: a changed
  default, or an existing code path changed in place. Say what changes for
  that run.
- Training, rollout, loss or checkpoint code changed without a test, or
  without a line in the description saying how it was verified on Thur (a
  command or a job id).
- A resumed run that would not continue its first attempt: checkpoint, data
  position, RNG.
- A key or token committed to the repository.

Leave out:

- In a cherry-pick of an upstream pull request, everything identical to the
  upstream commit. Review what differs.
- Code the pull request did not touch, and what `.pre-commit-config.yaml`
  enforces.

Each finding names the file and line and the concrete case in which the code
goes wrong, with the fix when it is short. At most 10 per review, the most
severe first; report nothing rather than pad. A review by an agent is not an
approval: the author fixes or answers each finding and resolves the
conversation.
