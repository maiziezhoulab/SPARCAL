import configparser
import os

# Define the base directory and the template file path
snv_tool = 'self'
base_dir = f'/data/maiziezhou_lab/hanliu/projects/snv_call/configs/{snv_tool}/'
template_file = os.path.join(base_dir, '151507.ini')

# Define the IDs to create configurations for
ids = ['151508', '151509', '151510'] + [f'15166{i}' for i in range(9, 10)] + [f'15167{i}' for i in range(0, 7)]

def create_config_files(template_file, base_dir, ids):
    # Read the template INI file
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(template_file)

    for spatial_id in ids:
        # Create a new configuration instance
        new_config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

        # Copy the DEFAULT section
        new_config['DEFAULT'] = config['DEFAULT']

        # Modify the spatialid in the config
        new_config['DEFAULT']['spatialid'] = spatial_id
        new_config['DEFAULT']['local_data_path'] = f"/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/{spatial_id}/{snv_tool}"
        new_config['DEFAULT']['result'] = f"/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/{spatial_id}/{snv_tool}/results/{new_config['DEFAULT']['filter']}/{new_config['DEFAULT']['neighbor_lim']}{new_config['DEFAULT']['filter_threshold']}"

        new_config['INPUT'] = {
            'tissue_pos_path': f"{new_config['DEFAULT']['maizie_path']}/{spatial_id}/tissue_positions.csv",
            'bam_path': f"{new_config['DEFAULT']['maizie_path']}/{spatial_id}/bam_bycell"
        }

        new_config['PROCESS'] = {
            'neighbor_pixel_path': f"{new_config['DEFAULT']['local_data_path']}/{new_config['DEFAULT']['neighbor_lim']}.csv",
            'merged_bam_path': f"{new_config['DEFAULT']['maizie_path']}/{spatial_id}/bam_bycell",
            'out_vcf_path': f"{new_config['DEFAULT']['local_data_path']}/output_VCFs/{new_config['DEFAULT']['filter']}/{new_config['DEFAULT']['neighbor_lim']}{new_config['DEFAULT']['filter_threshold']}/",
            'vcf_tables_path': f"{new_config['DEFAULT']['local_data_path']}/processed_data/vcf_tables/{new_config['DEFAULT']['filter']}/{new_config['DEFAULT']['neighbor_lim']}{new_config['DEFAULT']['filter_threshold']}/",
            'processed_vcf_tables_path': f"{new_config['DEFAULT']['local_data_path']}/processed_data/processed_vcf_tables/{new_config['DEFAULT']['filter']}/{new_config['DEFAULT']['neighbor_lim']}{new_config['DEFAULT']['filter_threshold']}/"
        }

        new_config['RESULT'] = {
            'union_set_path': f"{new_config['DEFAULT']['result']}/union_set.csv",
            'matrix_path': f"{new_config['DEFAULT']['result']}/matrix.pkl",
            'modified_matrix_path': f"{new_config['DEFAULT']['result']}/all_inTissue_matrix.pkl",
            'sum_snp_path': f"{new_config['DEFAULT']['result']}/sum_snp.txt",
            'histogram_path': f"{new_config['DEFAULT']['result']}/histogram.png",
            'cluster_path': f"{new_config['DEFAULT']['result']}/clusters_{new_config['DEFAULT']['num_cluster']}.csv",
            'visual_path': f"{new_config['DEFAULT']['result']}/clusters_{new_config['DEFAULT']['num_cluster']}.png",
            'mclust_result': f"{new_config['DEFAULT']['local_data_path']}/results/{new_config['DEFAULT']['filter']}/mclust_result/mclust.csv"
        }

        new_config['TBSP'] = {
            'groupcell_path': f"{new_config['DEFAULT']['local_data_path']}/tbsp/{new_config['DEFAULT']['neighbor_lim']}/GroupCells.txt",
            'tbsp_cluster_path': f"{new_config['DEFAULT']['local_data_path']}/results/{new_config['DEFAULT']['neighbor_lim']}/tbsp_clusters_{new_config['DEFAULT']['num_cluster']}.csv",
            'tbsp_visual_path': f"{new_config['DEFAULT']['local_data_path']}/results/{new_config['DEFAULT']['neighbor_lim']}/tbsp_clusters_{new_config['DEFAULT']['num_cluster']}.png"
        }

        # Write the new config file
        output_file = os.path.join(base_dir, f'{spatial_id}.ini')
        with open(output_file, 'w') as configfile:
            new_config.write(configfile)
        print(f"Config file created: {output_file}")

# Create config files for all specified IDs
create_config_files(template_file, base_dir, ids)
