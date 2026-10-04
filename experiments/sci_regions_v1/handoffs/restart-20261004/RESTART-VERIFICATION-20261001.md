# RESTART VERIFIED — SCI-REGIONS

Fresh direct verification: 2026-10-01T09:28:28.026068+00:00.
Scope: this SCI-REGIONS chat only. No development, feature merge, deployment, service or experiment was started. Manual-test freeze remains in force. Lane remains stood down.

## Actual owned Git state
- Inventory discovered the retained historical checkout at /Users/paulslee/Documents/GitHub/SCI-REGIONS-v1-local-snapshot/committed-branch; it is the only Git checkout in that owned snapshot tree. git worktree list shows only itself.
- git status --short --untracked-files=all returned no entries. HEAD is e3ddf72d86ca881bf770ee4ec5b09852ce8912df on sci/regions-v1-prototype. git fsck --full completed successfully.
- Fresh git ls-remote reports sci/regions-v1-prototype at cef7f7c66f1653d5006194f297fbb1eda6c14392 and sci/regions-contrastive-vulnerability at 68e8c8874eed528422db186852d3a5a1da9a27da.
- GitHub compare directly confirms historical local e3ddf72 is the merge base, 4 commits behind remote cef7, and zero commits ahead. This checkout is an intentionally older snapshot; it has no unpushed or uncommitted work. It was not advanced during the freeze.
- The newer standalone sci-regions-coordinate-labels checkout is absent. Its complete-history Git bundle has both exact current remote heads and HEAD at 68e8c887. Fresh git bundle verify succeeded using the retained checkout; git bundle list-heads and the full bundle checksum matched.
- App list_artifacts returned no attached managed worktrees or PRs.

## Durable evidence and context
- All eight entries in this directory's SHA256SUMS verified; every regular checkpoint file was read in full.
- All 1,335 older snapshot checksums verified; all 1,381 regular files including Git internals were readable.
- sci-owned-source.tar.gz fully read: 88 committed source/evidence files. Its results.zip passed all-member ZIP integrity verification (27 entries). The three separately banked JSON/coaching files are byte-identical to their archive copies.
- Scripts, schemas, frozen fixtures, independent oracles, source corpus/mappings, raw results, comparison HTML, manifests, mutant evidence and tests are present in the archive and/or older snapshot. HANDOFF.md, STOOD-DOWN-20261001.md and RESTART-HANDOFF.md are readable. Older handoffs record historical state; this receipt records the fresh exception above.
- The thread-specific visualization directory contains no files (directory absent). No visualization depends on temporary storage.
- /private/tmp name inventory found an empty ai-experience-contrastive-witness directory and two Build-authored historical result messages. The two messages have been copied byte-for-byte into retained-context/ here, without deleting or modifying their originals. No unique SCI restart material remains dependent on /private/tmp.

## Runtime
- Direct process inventory found no SCI-owned watcher, evaluator, test, browser runner, server, PTY or background job. Only the verification commands themselves appeared.
- Agent inventory reports only root running this verification, with no child agent in flight.
- No automation.toml matches this thread or the two former SCI monitor IDs. No monitor was started.
- No other owner's services or checkouts were stopped or modified.

## Resume
Read this receipt first, then STOOD-DOWN-20261001.md and HANDOFF.md. Complete local recovery is from sci-repository.bundle; frozen source-only access is sci-owned-source.tar.gz. Do not resume feature work without the official named lane restart described in the standdown handoff. No production deployment is claimed.

Exception: the older clean snapshot is four commits behind its remote; the complete newer heads are saved locally in the verified bundle and pushed. This is not unpushed work or a restart blocker.
