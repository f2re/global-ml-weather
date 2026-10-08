"""Synthetic fixed-width fixtures; no synthetic profile is a real archive result."""
from datetime import timedelta
import gzip
import io
from pathlib import Path
import zipfile

import pytest

from global_weather.providers import igra
from global_weather.pipeline.io import sha256


STATION='USM00072501'


def header(*,year=2021,hour=12,release=1130,count=1,station=STATION):
    line=list(' '*71)
    fields=[(1,1,'#'),(2,12,station),(14,17,f'{year:04d}'),(19,20,'01'),(22,23,'02'),
            (25,26,f'{hour:02d}'),(28,31,f'{release:04d}'),(33,36,f'{count:4d}'),
            (38,45,'ncdc-gts'),(47,54,'ncdc-gts'),(56,62,f'{400000:7d}'),(64,71,f'{-750000:8d}')]
    for first,last,text in fields:line[first-1:last]=text
    return ''.join(line)+'\n'


def level(*,pressure=50000,temperature=-100,elapsed=130,height=5600,rh=500,dpdp=100,
          direction=90,speed=100,pflag='B',tflag='B',zflag='A'):
    line=list(' '*51)
    fields=[(1,1,'1'),(2,2,'0'),(4,8,f'{elapsed:5d}'),(10,15,f'{pressure:6d}'),
            (16,16,pflag),(17,21,f'{height:5d}'),(22,22,zflag),(23,27,f'{temperature:5d}'),
            (28,28,tflag),(29,33,f'{rh:5d}'),(35,39,f'{dpdp:5d}'),
            (41,45,f'{direction:5d}'),(47,51,f'{speed:5d}')]
    for first,last,text in fields:line[first-1:last]=text
    return ''.join(line)+'\n'


def text_source(tmp_path,text):
    path=tmp_path/(STATION+'-data.txt');path.write_text(text,encoding='ascii');return path


def normalized(profile):
    return igra.normalized_records(profile,acquired_at='2026-10-08T12:00:00Z',
                                   archive_sha256='a'*64,format_sha256='b'*64)


def test_actual_elapsed_units_qc_and_sparse_variables(tmp_path):
    path=text_source(tmp_path,header()+level())
    profile=list(igra.iter_profiles(path))[0];records=normalized(profile)
    values={r['variable']:r for r in records}
    assert values['temperature']['value']==pytest.approx(263.15)
    assert values['temperature']['observed_at']=='2021-01-02T11:31:30+00:00'
    assert values['temperature']['available_at']=='2021-01-02T13:31:30+00:00'
    assert values['u']['value']==pytest.approx(-10.)
    assert values['v']['value']==pytest.approx(0.,abs=1e-12)
    assert values['geopotential']['value']==pytest.approx(5600*9.80665)
    assert 0<values['specific_humidity']['value']<.2
    assert 'omega' not in values
    assert all(r['profile_id']==records[0]['profile_id'] for r in records)
    assert all(r['provider_qc']['PFLAG']=='B' for r in records)
    assert all(r['actual_available_at'] is None for r in records)
    assert all(r['position_basis']=='launch_position_fallback' for r in records)


def test_missing_and_qc_flags_preserve_other_variables(tmp_path):
    path=text_source(tmp_path,header(count=2)+level(temperature=-8888,height=-9999,rh=-9999,dpdp=-9999)
                     +level(temperature=200,tflag='X',pressure=70000))
    records=normalized(list(igra.iter_profiles(path))[0])
    assert all(r['variable']!='temperature' for r in records)
    assert any(r['variable']=='u' for r in records)
    assert any(r['variable']=='geopotential' for r in records)
    path=text_source(tmp_path,header()+level(pflag='X'))
    assert list(igra.iter_profiles(path))==[]
    path=text_source(tmp_path,header()+level(speed=0,direction=-9999,temperature=-9999,height=-9999))
    records=normalized(list(igra.iter_profiles(path))[0])
    assert {record['variable']:record['value'] for record in records}=={'u':0.,'v':0.}


def test_undocumented_surface_wind_only_row_is_quarantined_without_losing_profile(tmp_path):
    # Exact observed NOAA pattern; code 0 is not declared supported by the format.
    raw='01 -9999  -9999 -9999 -9999 -9999 -9999    90    41 '
    from collections import Counter
    report=Counter()
    path=text_source(tmp_path,header(count=2)+raw+'\n'+level())
    profile=list(igra.iter_profiles(path,report=report))[0]
    records=normalized(profile)
    assert len(profile['levels'])==1
    assert {record['variable'] for record in records}=={'temperature','specific_humidity','u','v','geopotential'}
    assert all(record['pressure_pa']==50000 for record in records)
    assert report['quarantined_undocumented_01_surface_without_vertical_coordinate']==1
    assert report['omitted_levels']==1 and report['profiles']==1
    # A real pressure, QA-removed field or another unknown type cannot be coerced.
    invalid=[ '01'+level()[2:], raw[:9]+f'{50000:6d}'+raw[15:],
              raw[:16]+f'{-8888:5d}'+raw[21:], '41'+raw[2:] ]
    for row in invalid:
        path=text_source(tmp_path,header()+row+'\n')
        with pytest.raises(ValueError,match='Invalid IGRA level type'):
            list(igra.iter_profiles(path))


def test_release_partial_minute_and_midnight_are_explicit(tmp_path):
    path=text_source(tmp_path,header(release=1199)+level())
    profile=list(igra.iter_profiles(path))[0];records=normalized(profile)
    assert profile['launch_time'] is None
    assert all(r['time_basis']=='nominal_time_fallback' for r in records)
    assert all('elapsed_time_without_precise_release' in r['omitted_igra_fields'] for r in records)
    path=text_source(tmp_path,header(hour=0,release=2330)+level())
    profile=list(igra.iter_profiles(path))[0]
    assert profile['launch_time']=='2021-01-01T23:30:00+00:00'
    assert profile['release_date_basis']=='nearest_nominal_day_inferred'


def test_whole_profile_split_boundary_guard():
    train=[{'observed_at':(igra.TRAIN_END-timedelta(hours=43)).isoformat()}]
    validation=[{'observed_at':(igra.TRAIN_END+timedelta(hours=42)).isoformat()}]
    assert igra.group_split(train)=='train'
    assert igra.group_split(validation)=='validation'
    assert igra.group_split(train+validation) is None
    assert igra.group_split([{'observed_at':igra.TRAIN_END.isoformat()}]) is None


def test_train_admission_pass_does_not_parse_holdout_measurements(tmp_path):
    path=text_source(tmp_path,header()+level()+header(year=2022,release=9900)+'not-a-measurement\n')
    assert len(list(igra.iter_profiles(path,years=(2021,))))==1
    with pytest.raises(ValueError):list(igra.iter_profiles(path,years=(2021,2022)))


def test_compressed_archives_are_streamed_and_bounded(tmp_path,monkeypatch):
    text=header()+level();raw=text.encode()
    gz=tmp_path/'source.gz'
    with gzip.open(gz,'wb') as stream:stream.write(raw)
    assert len(list(igra.iter_profiles(gz)))==1
    zip_path=tmp_path/'source.zip'
    with zipfile.ZipFile(zip_path,'w') as archive:archive.writestr(STATION+'-data.txt',raw)
    assert len(list(igra.iter_profiles(zip_path)))==1
    monkeypatch.setattr(igra,'MAX_EXPANDED',len(raw)-1)
    with pytest.raises(ValueError,match='bounds'):
        list(igra.iter_profiles(gz))
    with pytest.raises(ValueError,match='Unsafe'):
        list(igra.iter_profiles(zip_path))


@pytest.mark.parametrize('member',['../USM00072501-data.txt','dir/USM00072501-data.txt','unexpected.bin'])
def test_zip_never_extracts_or_accepts_unexpected_members(tmp_path,member):
    path=tmp_path/'unsafe.zip'
    with zipfile.ZipFile(path,'w') as archive:archive.writestr(member,header()+level())
    with pytest.raises(ValueError,match='Unsafe'):
        list(igra.iter_profiles(path))


def test_station_source_identity_and_truncation(tmp_path):
    with pytest.raises(ValueError):igra.station_url('../../etc/passwd')
    path=text_source(tmp_path,header()+level())
    with pytest.raises(ValueError,match='identity mismatch'):
        list(igra.iter_profiles(path,expected_station='CAM00071109'))
    path=text_source(tmp_path,header(count=2)+level())
    with pytest.raises(ValueError,match='Truncated'):
        list(igra.iter_profiles(path))
    with pytest.raises(ValueError,match='1–128'):
        igra.acquire(tmp_path/'cache',tmp_path/'out',station_limit=129)


def test_inventory_filters_only_train_period_and_diversifies_cells(tmp_path):
    def station(identity,lat,lon,first,last):
        line=list(' '*88)
        for start,end,text in [(1,11,identity),(13,20,f'{lat:8.4f}'),(22,30,f'{lon:9.4f}'),
                               (32,37,f'{123.:6.1f}'),(42,71,'TEST'.ljust(30)),
                               (73,76,f'{first:4d}'),(78,81,f'{last:4d}'),(83,88,'999999')]:
            line[start-1:end]=text
        return ''.join(line)+'\n'
    path=tmp_path/'inventory.txt'
    path.write_text(station(STATION,40,-75,1950,2021)+station('CAM00071109',50,-100,2022,2026))
    inventory=igra.station_inventory(path)
    assert [row['id'] for row in inventory]==[STATION]
    assert 'nobs' not in inventory[0]  # Whole-record count never ranks candidates.
    rows=[{'id':f'USM{i:08d}','latitude':lat,'longitude':lon} for i,(lat,lon) in
          enumerate([(0,0),(0,1),(40,100),(-40,-100)])]
    candidates=igra.spatial_candidates(rows)
    assert len(candidates)==4
    assert candidates[-1]['id']==rows[1]['id']


def test_inventory_utf8_name_keeps_character_columns_and_strict_structural_fields(tmp_path):
    # Synthetic fixed-width reconstruction of the names found in NOAA's cache.
    name='BEOGRAD/KOŠUTNJAK'
    line=list(' '*88)
    for start,end,text in [(1,11,'RIM00013275'),(13,20,' 44.7714'),(22,30,'  20.4244'),
                           (32,37,' 203.0'),(42,71,name.ljust(30)),
                           (73,76,'1971'),(78,81,'2026'),(83,88,' 40125')]:
        line[start-1:end]=text
    text=''.join(line)
    assert len(text)==88 and len(text.encode('utf-8'))==89
    path=tmp_path/'inventory.txt';path.write_text(text+'\n',encoding='utf-8')
    result=igra.station_inventory(path)
    assert result==[{'id':'RIM00013275','latitude':44.7714,'longitude':20.4244,
                    'elevation_m':203.,'name':name,'first_year':1971,'last_year':2026}]
    # Unicode decimal numerals must never silently extend the numeric format.
    malformed=text[:72]+'١'+text[73:]
    path.write_text(malformed+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='outside the station name'):
        igra.station_inventory(path)
    path.write_bytes((text+'\n').encode('cp1250'))
    with pytest.raises(UnicodeDecodeError):igra.station_inventory(path)


def test_inventory_quarantines_only_observed_anonymous_row_pattern(tmp_path):
    # The observed official-cache row has no ID, coordinates or name, only a tail.
    anonymous=' '*72+'1946 2025  70410'
    assert len(anonymous)==88
    line=list(' '*88)
    for start,end,text in [(1,11,STATION),(13,20,' 40.0000'),(22,30,' -75.0000'),
                           (32,37,' 123.0'),(42,71,'TEST'.ljust(30)),
                           (73,76,'2021'),(78,81,'2021'),(83,88,'000001')]:
        line[start-1:end]=text
    valid=''.join(line)
    path=tmp_path/'inventory.txt';path.write_text(anonymous+'\n'+valid+'\n',encoding='utf-8')
    report={};stations=igra.station_inventory(path,report=report)
    assert [station['id'] for station in stations]==[STATION]
    assert report=={'anonymous_catalog_rows':1,'quarantined_rows':[{'line_number':1,
        'reason':'blank_station_identity_and_geometry','first_year':1946,'last_year':2025,
        'reported_sounding_count':70410}]}
    for bad in (anonymous[:72]+'oops'+anonymous[76:],
                'BAD'.ljust(11)+anonymous[11:],
                valid[:12]+' '*8+valid[20:]):
        path.write_text(bad+'\n',encoding='utf-8')
        with pytest.raises(ValueError):igra.station_inventory(path)


def test_acquire_with_synthetic_provider_and_resume_hash_gate(tmp_path,monkeypatch):
    line=list(' '*88)
    for start,end,text in [(1,11,STATION),(13,20,' 40.0000'),(22,30,' -75.0000'),
                           (32,37,' 123.0'),(42,71,'TEST'.ljust(30)),
                           (73,76,'2021'),(78,81,'2021'),(83,88,'000001')]:
        line[start-1:end]=text
    catalog=(''.join(line)+'\n').encode()
    compressed=io.BytesIO()
    with zipfile.ZipFile(compressed,'w') as archive:
        archive.writestr(STATION+'-data.txt',header()+level())
    requested=[]
    def fake_download(url,path,*,max_bytes,network):
        assert url.startswith(igra.BASE) and network is False
        requested.append(url)
        payload=(catalog if url.endswith('igra2-station-list.txt') else
                 compressed.getvalue() if url.endswith('.zip') else b'synthetic format fixture')
        assert len(payload)<=max_bytes
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        return {'url':url,'sha256':sha256(path),'bytes':len(payload),
                'acquired_at':'2026-10-08T12:00:00Z'}
    monkeypatch.setattr(igra,'download',fake_download)
    output=tmp_path/'output';cache=tmp_path/'cache'
    manifest=igra.acquire(cache,output,station_limit=1,minimum_train_profiles=1)
    assert manifest['provider']=='NOAA_IGRA2' and manifest['omega_supported'] is False
    assert manifest['stations'][0]['train_profiles']==1
    assert manifest['stations'][0]['source_qc']['profiles']==1
    assert (output/'observations.jsonl').is_file()
    calls=len(requested)
    assert igra.acquire(cache,output,station_limit=1,minimum_train_profiles=1)==manifest
    assert len(requested)==calls
    with (output/'observations.jsonl').open('a') as stream:stream.write('\n')
    with pytest.raises(ValueError,match='changed'):
        igra.acquire(cache,output,station_limit=1,minimum_train_profiles=1)
