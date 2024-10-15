import os
import pandas as pd
import haven.db as db

if __name__ == '__main__':
    os.environ['HAVEN_DATABASE'] = 'haven'

    tags = pd.read_csv('tags.csv.gz')
    tags['tag_key'] = tags['tag_key'].astype('str')
    tags['upload_key'] = 'mgietzmann'

    db.write_data(
        tags, 'mgietzmann_tags', 
        ['upload_key']
    )

    tag_tracks = pd.read_csv('tag_tracks.csv.gz')
    tag_tracks['tag_key'] = tag_tracks['tag_key'].astype('str')
    tag_tracks['upload_key'] = 'mgietzmann'

    db.write_data(
        tag_tracks, 'mgietzmann_tag_tracks', 
        ['upload_key']
    )

    tag_depths = pd.read_csv('tag_depths.csv.gz')
    tag_depths['tag_key'] = tag_depths['tag_key'].astype('str')
    tag_depths['upload_key'] = 'mgietzmann'

    db.write_data(
        tag_depths, 'mgietzmann_tag_depths', 
        ['upload_key']
    )