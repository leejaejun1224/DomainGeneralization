#!/usr/bin/env python3
# file: gather_files_from_list.py

import argparse, os, sys, shutil
from pathlib import Path
from typing import List, Tuple

def read_list(txt_path: Path) -> List[str]:
    lines = []
    with txt_path.open('r', encoding='utf-8') as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            lines.append(s)
    # 중복 제거(원본 순서 유지)
    seen, out = set(), []
    for s in lines:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def find_one(target: str, root: Path) -> List[Path]:
    """
    target이 경로처럼 보이면 root와 결합해 우선 확인.
    없으면 basename으로 root 하위 전체를 탐색.
    """
    # 경로 normalize (슬래시/백슬래시 혼용 대비)
    norm_target = os.path.normpath(target)
    cand = (root / norm_target) if not os.path.isabs(norm_target) else Path(norm_target)
    hits: List[Path] = []
    if cand.exists():
        hits.append(cand.resolve())
        return hits

    # basename으로 재귀 검색
    base = Path(norm_target).name
    # 성능: 동일 basename이 아주 많다면 필요 시 여기서 확장자 필터 추가 가능
    for p in root.rglob(base):
        if p.is_file() and p.name == base:
            hits.append(p.resolve())
    return hits

def safe_write_path(dest_dir: Path, rel_path: Path, preserve_structure: bool) -> Path:
    if preserve_structure:
        out_path = dest_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
        base = rel_path.name
        out_path = dest_dir / base
        if not out_path.exists():
            return out_path
        stem, suf = Path(base).stem, Path(base).suffix
        k = 1
        while True:
            c = dest_dir / f"{stem}_{k}{suf}"
            if not c.exists():
                return c
            k += 1

def transfer(src: Path, dst: Path, mode: str):
    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'move':
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    elif mode == 'symlink':
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    ap = argparse.ArgumentParser(description="txt 파일 목록에 있는 파일만 모아 폴더로 수집")
    ap.add_argument('--txt', default="/home/jaejun/DomainGeneralization/filenames/source/flyingthing_train copy.txt", type=Path)
    ap.add_argument('--root', default="/home/jaejun/dataset/flyingthing/")
    ap.add_argument('--dest', default="./collected_files")
    ap.add_argument('--mode', choices=['copy','move','symlink'], default='copy', help='수집 방식')
    ap.add_argument('--preserve-structure', action='store_true', help='원 경로 구조 보존(루트 기준 상대경로로 재현)')
    ap.add_argument('--strict', action='store_true', help='목록의 항목이 하나도 안 맞으면 에러로 종료')
    args = ap.parse_args()

    args.txt  = Path(args.txt).expanduser()
    args.root = Path(args.root).expanduser()
    args.dest = Path(args.dest).expanduser()

    lst = read_list(args.txt)
    if not lst:
        print("txt에 유효한 항목이 없습니다.", file=sys.stderr); sys.exit(1)
    root = args.root.resolve()
    dest = args.dest.resolve()
    if not root.exists():
        print(f"[오류] root가 존재하지 않음: {root}", file=sys.stderr); sys.exit(1)

    total_hits, missing = 0, []
    transfers: List[Tuple[Path, Path]] = []

    for line in lst:
        hits = find_one(line, root)
        if not hits:
            missing.append(line)
            continue
        for src in hits:
            try:
                rel = src.relative_to(root)
            except ValueError:
                # root 밖의 절대경로가 들어온 경우: 구조 보존 시 드라이브/루트 제거
                rel = Path(src.name) if not args.preserve_structure else Path(*src.parts[1:])
            dst = safe_write_path(dest, rel, args.preserve_structure)
            transfers.append((src, dst))
            total_hits += 1

    # 실제 전송
    for src, dst in transfers:
        transfer(src, dst, args.mode)

    print(f"[완료] 수집 방식: {args.mode}")
    print(f" - 총 요청 항목: {len(lst)}")
    print(f" - 매칭된 파일 수: {total_hits}")
    print(f" - 목적지: {dest}")
    if missing:
        print(f" - 매칭 실패 {len(missing)}건(예시 최대 10건):")
        for m in missing[:10]:
            print(f"   · {m}")
        if args.strict:
            sys.exit(2)

if __name__ == "__main__":
    main()
