from datetime import datetime

import yaml
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean, and_, Interval
from sqlalchemy import Sequence
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

credentials = yaml.load(open("../../credentials.yaml"), Loader=yaml.FullLoader)

engine = create_engine(
    f'postgresql://{credentials["username"]}:{credentials["password"]}@{credentials["url"]}:{credentials["port"]}/slurm-info',
    echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)


class Cluster(Base):
    __tablename__ = 'cluster'
    id = Column(Integer, Sequence('id'), primary_key=True)

    # references (not really necessary so far...)
    # cluster_resources = relationship("ClusterResources", backref="cluster")

    # attributes
    name = Column(String(50), nullable=False)
    number_of_gpus = Column(Integer(), nullable=False)
    total_memory_gpus = Column(Integer(), nullable=False)
    total_memory = Column(Integer(), nullable=False)
    total_cpu_cores = Column(Integer(), nullable=False)

    def __repr__(self):
        return f"<Cluster(name='{self.name}', number_of_total_gpus='{self.number_of_gpus}', total_gpu_memory='{self.total_memory_gpus}', total_memory='{self.total_memory}', total_cpu_cores='{self.total_cpu_cores}'')>"


class ClusterResources(Base):
    __tablename__ = 'cluster_resources'
    id = Column(Integer, Sequence('id'), primary_key=True)

    # references
    cluster_id = Column(Integer, ForeignKey("cluster.id"),
                        nullable=False)  # we store cluster id and not cluster resource id as cluster resrouces are changing every minute while slurm jobs are constant over time
    # slurm_processes = relationship("SlurmProcess", backref="cluster_resources")  # (not really necessary so far...)

    # attributes
    is_active = Column(Boolean(), nullable=False)
    timestamp = Column(DateTime(), nullable=False)
    total_memory_used_gpus = Column(Integer(), nullable=False)
    total_memory_reserved = Column(Integer(), nullable=False)
    total_cpu_cores_reserved = Column(Integer(), nullable=False)
    n_busy_gpus = Column(Integer(), nullable=False)
    n_reserved_gpus = Column(Integer(), nullable=False)
    n_containers = Column(Integer(), nullable=False)
    n_slurm_jobs = Column(Integer(), nullable=False)

    def __repr__(self):
        return f"<ClusterResources(used_memory='{self.total_memory_used_gpus}', total_memory_reserved='{self.total_memory_reserved}', total_cpu_cores_reserved='{self.total_cpu_cores_reserved}', n_busy_gpus='{self.n_busy_gpus}', n_reserved_gpus='{self.n_reserved_gpus}', n_containers='{self.n_containers}', n_slurm_jobs='{self.n_slurm_jobs}'')>"


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, Sequence('id'), primary_key=True)

    # references (not really necessary so far...)
    # slurm_processes = relationship("SlurmProcess", backref="user")
    # messages = relationship("Message", backref="user")

    # attributes
    abbreviation = Column(String(50), nullable=False, unique=True)
    email = Column(String(100), nullable=False, unique=True)

    def __repr__(self):
        return f"<User(name='{self.abbreviation}', email='{self.email}')>"


class SlurmProcess(Base):
    __tablename__ = 'slurm_processes'
    id = Column(Integer, Sequence('id'), primary_key=True)

    # references
    cluster_id = Column(Integer, ForeignKey("cluster.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # attributes
    status = Column(String(50))
    runtime = Column(Interval())
    n_gpus = Column(Integer())
    job_id = Column(Integer())
    n_cpus = Column(Integer())
    min_memory = Column(Integer())
    is_interactive_bash = Column(Boolean())
    not_using_gpu = Column(Boolean())
    valid_from = Column(DateTime(), nullable=False)
    valid_until = Column(DateTime(), nullable=False,
                         default=datetime.strptime('31/12/9999 23:59:59', '%d/%m/%Y %H:%M:%S'))

    def __repr__(self):
        return f"<SlurmProcess(status='{self.status}', runtime='{self.runtime}', job_id='{self.job_id}', n_gpus={self.n_gpus}', n_cpus={self.n_cpus}', min_memory={self.min_memory}', is_interactive_bash='{self.is_interactive_bash}', not_using_gpu='{self.not_using_gpu}')>"


class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, Sequence('id'), primary_key=True)

    # references
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    cluster_id = Column(Integer, ForeignKey("cluster.id"), nullable=False)

    # attributes
    severity = Column(Integer)
    message = Column(String(500))
    data = Column(JSONB)
    valid_from = Column(DateTime(), nullable=False)
    valid_until = Column(DateTime(), nullable=False,
                         default=datetime.strptime('31/12/9999 23:59:59', '%d/%m/%Y %H:%M:%S'))

    def __repr__(self):
        return f"<Message(severity='{self.severity}', message='{self.message}', user_id='{self.user_id}')>"


def setup_tables():
    Base.metadata.create_all(engine)


def insert_clusters():
    session = Session()
    clusters_to_add = []
    for name, n_gpus, total_memory_gpus, total_memory, total_cpu_cores in [("DGX-1", 8, 272, 512, 80),
                                                                           ("DGX-2", 8, 272, 512, 80),
                                                                           ("DGX-3", 8, 272, 512, 80),
                                                                           ("DGX-A100", 8, 340, 1024, 256)]:
        if len(session.query(Cluster).filter(Cluster.name.in_([name])).all()) <= 0:
            clusters_to_add.append(Cluster(name=name, number_of_gpus=n_gpus, total_memory_gpus=total_memory_gpus,
                                           total_memory=total_memory, total_cpu_cores=total_cpu_cores))

    if len(clusters_to_add) > 0:
        session.add_all(clusters_to_add)
        session.commit()

    session.close()


def get_cluster_by_name(name):
    session = Session()
    cluster = session.query(Cluster).filter_by(name=name).first()
    session.close()
    return cluster


def get_active_messages(cluster_id):
    session = Session()
    active_messages = session.query(Message).filter(and_(
        Message.valid_until >= datetime.strptime('31/12/9999 23:59:59', '%d/%m/%Y %H:%M:%S'),
        Message.cluster_id == cluster_id)).all()
    session.close()
    return active_messages


def get_active_slurm_jobs(cluster_id):
    session = Session()
    active_jobs = session.query(SlurmProcess).filter(
        and_(SlurmProcess.status == "RUNNING", SlurmProcess.cluster_id == cluster_id,
             SlurmProcess.valid_until >= datetime.strptime('31/12/9999 23:59:59', '%d/%m/%Y %H:%M:%S'))).all()
    session.close()
    return active_jobs


def get_users_abbreviation_dict():
    session = Session()
    users = session.query(User).all()
    session.close()

    users_dict = {}
    for user in users:
        users_dict[user.abbreviation] = user

    return users_dict


def add_objects(objects: list):
    if len(objects) > 0:
        session = Session()
        session.add_all(objects)
        session.commit()
        session.close()


def add_cluster_resources(cluster_resources):
    session = Session()
    if cluster_resources.is_active:
        session.query(ClusterResources).filter(
            and_(ClusterResources.is_active, ClusterResources.cluster_id == cluster_resources.cluster_id)). \
            update({"is_active": False}, synchronize_session="fetch")
    session.add(cluster_resources)
    session.commit()
    session.refresh(cluster_resources)
    session.close()
    return cluster_resources


def update_messages_valid_date(msg_valid_dict):
    session = Session()
    for msg, date in msg_valid_dict.items():
        session.query(Message).filter(Message.id == msg.id). \
            update({"valid_until": date}, synchronize_session="fetch")
    session.commit()
    session.close()


def update_slurm_jobs(update_jobs_runtime, update_jobs_validity):
    session = Session()
    _update_slurm_jobs_runtime(session, update_jobs_runtime)
    _update_slurm_jobs_validity(session, update_jobs_validity)
    session.commit()
    session.close()


def _update_slurm_jobs_runtime(session, update_jobs_runtime):
    for job, runtime in update_jobs_runtime.items():
        session.query(SlurmProcess).filter(SlurmProcess.id == job.id). \
            update({"runtime": runtime}, synchronize_session="fetch")


def _update_slurm_jobs_validity(session, update_jobs_validity):
    for job, validity in update_jobs_validity.items():
        session.query(SlurmProcess).filter(SlurmProcess.id == job.id). \
            update({"valid_until": validity}, synchronize_session="fetch")


def _update_slurm_jobs_status(session, update_jobs_status):
    for job, status in update_jobs_status.items():
        session.query(SlurmProcess).filter(SlurmProcess.id == job.id). \
            update({"status": status}, synchronize_session="fetch")


def _update_slurm_jobs_gpu_usage(session, update_jobs_gpu_usage):
    for job, not_using_gpu in update_jobs_gpu_usage.items():
        session.query(SlurmProcess).filter(SlurmProcess.id == job.id). \
            update({"not_using_gpu": not_using_gpu}, synchronize_session="fetch")


if __name__ == "__main__":
    setup_tables()
    insert_clusters()
    session = Session()
    print(session.query(Cluster).all())
    session.close()
