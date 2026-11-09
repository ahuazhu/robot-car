"""Build a local Arduino ESP-NN library from a pinned, verified vendor archive."""
import argparse,hashlib,json,re,shutil,tarfile
from pathlib import Path

COMMIT='744d48deae6aa4a4d7b93b6a80f548d5298813a4'
SHA256='3a895a6d0f35d0eda8adbe368c34e0615825191c6d9e49411cb06cd4b910fbdd'
URL=f'https://codeload.github.com/espressif/esp-nn/tar.gz/{COMMIT}'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if hashlib.sha256(a.archive.read_bytes()).hexdigest()!=SHA256:
        raise SystemExit('ESP-NN archive SHA256 mismatch')
    a.out.mkdir(parents=True,exist_ok=True)
    unpack=a.out/'vendor';unpack.mkdir(exist_ok=True)
    with tarfile.open(a.archive) as tar:
        for item in tar.getmembers():
            target=(unpack/item.name).resolve()
            if not target.is_relative_to(unpack.resolve()) or item.issym() or item.islnk():
                raise SystemExit('Unexpected archive member')
        tar.extractall(unpack)
    vendor=unpack/('esp-nn-'+COMMIT)
    library=a.out/'libraries/RobotEspNN';src=library/'src';src.mkdir(parents=True,exist_ok=True)
    # Follow the vendor's common + ESP32-S3 source lists, excluding other ISAs.
    cmake=(vendor/'CMakeLists.txt').read_text().split('# ESP32-S31')[0]
    sources=re.findall(r'"(src/[^\"]+\.(?:c|S))"',cmake)
    for relative in sources:
        dest=src/Path(relative).relative_to('src');dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.suffix=='.c':
            # Arduino applies -Os globally. Keep the vendor C body byte-for-byte
            # and compile it through a per-translation-unit O2 wrapper.
            body=dest.with_suffix('.upstream.inc');shutil.copy2(vendor/relative,body)
            dest.write_text('#pragma GCC optimize("O2", "no-unroll-loops")\n#include "'+body.name+'"\n')
        else:shutil.copy2(vendor/relative,dest)
    for header in (vendor/'src').rglob('*.h'):
        dest=src/header.relative_to(vendor/'src');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(header,dest)
    for folder in [vendor/'include',vendor/'src/common']:
        for header in folder.glob('*.h'):shutil.copy2(header,src/header.name)
    shutil.copy2(vendor/'LICENSE',library/'LICENSE')
    (library/'library.properties').write_text('name=RobotEspNN\nversion=1.4.2\nauthor=Espressif Systems\nmaintainer=robot-car\nsentence=Pinned ESP32-S3 SIMD kernels for LampNet\nparagraph=Local build with O2 wrappers; upstream Apache-2.0 bodies unmodified.\ncategory=Data Processing\narchitectures=esp32\nincludes=esp_nn.h\n')
    manifest={'url':URL,'commit':COMMIT,'sha256':SHA256,'version':'v1.4.2','sources':sources,'library':str(library),'c_optimization':'O2, no-unroll-loops'}
    (a.out/'esp-nn-build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'library':str(library),'sources':len(sources),'commit':COMMIT}))

if __name__=='__main__':main()
