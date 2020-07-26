import re
import pandas as pd
from typing import Tuple, List, Set


CHAIR_TYPES = ['W', 'P', 'S', 'C']
VERTICAL_WALL_SYMBOLS = '|\/+'
HORIZONTAL_WALL_SYMBOLS = '-'
WALL_SYMBOLS = VERTICAL_WALL_SYMBOLS + HORIZONTAL_WALL_SYMBOLS


class PlanFormatError(Exception):
    """ Raised if plan is not formatted correctly, i.e.
        1.) it contains invalid characters or
        2.) lines have different lengths, i.e. the plan is not rectangular
    """
    pass


class ChairDataExtractor:
    line_length: int
    chairs: pd.DataFrame
    status: List[int]
    new_status: List[int]

    def extract_data_from_file(self, file: str):
        """
        Reads plan from file and identifies rooms and chairs
        :param file:
        """

        self.chairs = pd.DataFrame(columns=['room'] + CHAIR_TYPES)

        with open(file, 'r') as f:
            line = f.readline()
            self.line_length = len(line)
            self.status = [None for k in range(self.line_length)]

            while line:
                self._check_validity(line)
                self._process_line(line)

            self._perform_final_validation()

    def get_loaded_chair_data(self) -> pd.DataFrame:
        """
        Returns the loaded data after setting its index to be the room name.

        :return: loaded chair data with room name as index and CHAIR_TYPES as columns
        """
        return self.chairs.set_index('name')

    def _check_validity(self, line: str):
        """
        Raises an error if line in plan has wrong length or contains invalid characters
        Valid characters are
            .) a-z for naming of rooms,
            .) +|/\- for walls,
            .) blanks for empty space,
            .) () for opening and closing of room names
            .) PWSC for chair identification

        :param line of plan in input file

        :raises PlanFormatError:
        """

        # check for valid characters
        valid_characters = WALL_SYMBOLS + ' a-z()PWSC'
        valid_pattern = '[' + valid_characters + ']+$'
        if not re.match(valid_pattern, line):
            raise PlanFormatError('Plan contains invalid characters in the following line:\n' + line)

        # check for uniform length in plan
        if len(line) != self.line_length:
            raise PlanFormatError('Plan contains lines of different lengths.')

    def _process_line(self, line: str) -> None:
        """
        Processes one line of the plan, identifies rooms and chairs; the processing works as follows:
            status encodes the rooms in the previous line - the k-th element of status will be:
                .) None if position N did not belong to any room
                .) integer if position k in the last line belonged to room k, where k is the index of the respective
                    room in chairs

        :param line: one line of the plan
        """

        self.new_status = [-1 if char in WALL_SYMBOLS else None for char in len(self.status)]
        room_sections = self._detect_room_sections(line)
        for section in room_sections:
            room = self._map_section_to_room(section, line)
            if room is not None:
                self._update_room_name_and_chairs(section, line, room)

        self.status = self.new_status
        self.new_status = None

    def _detect_room_sections(line: str) -> List[Tuple[int, int]]:
        """
        Detects all potential sections of a room in a line of the input file, i.e. all strings that start and end with
        a wall symbol, and contain non-wall-symbols in between
        :param line: string that should be searched for room sections
        :return: list of start and end points of room sections in line
        """

        room_section_pattern = '[|+/\-][^[|+/\-]]+[|+/\-]'

        found_sections = []
        room_section = re.search(room_section_pattern, line)
        while room_section:
            found_sections.append(room_section.span())
        return found_sections


    def _map_section_to_room(self, section: Tuple[int, int], line:str) -> int:
        """
        Map the the current section against room mapping of the previous line to determine, to which room
        it belongs.
        Three situations are possible:
            - The section is not mapped to any open room. In this case open a new room for it.
            - The section is mapped to one room only. Everything is okay.
            - The section is mapped to several rooms. This happens if the geometry of the flat is as follows:

                1    +-----------------------+
                2    |           |           |
                3    |           +           |
                4    |                       |
                5    +-----------------------+

            and we are currently in line 4. In this case two components the that have not been connected were
            tracked as different rooms so far. We correct this and join the two rooms in the dataframe chairs.

        :param section: tuple consisting of start and endpoint of room section in line
        :param line: line of input data
        :param new_status: room mapping status of the current line that needs to be updated

        :return: updated room mapping of the current line that incorporates the mapping of the section
        """
        # extract segment from line, ignore corners that are walls
        section_idx = slice(section[0] + 1, section[1])
        line_segment = line[section_idx]

        # map section against the open rooms stored in status, exclude wall symbols denoted by -1
        segment_status = self.status[section_idx]
        distinct_mapped_rooms = set([k for k in segment_status if k != -1 ])

        if distinct_mapped_rooms == {None}:
            if line == ' ' * len(line_segment):
                # we leave the section mapped to None
                room = None
            else:
                # if we are not in a room, but the the section contains other symbols than blanks,
                # the room plan contains errors. This can occur in situations like this:
                #
                #  1  +--+       +--+     or       +--+             +--+
                #  2  |  |  W S  |  |              |  |   (room 1)  |  |
                #  3  |  +-------+  |              |  +-------------+  |
                #  4  |             |              |                   |
                #  5  +-------------+              +-------------------+

                raise PlanFormatError('The plan contains rooms that are not fully enclosed by walls.')
        elif None in distinct_mapped_rooms:
            # In this case we detected a paritally open room as in the following example in line 2
            #
            #      1  +--+
            #      2  |  + - -   -+
            #      3  |           |
            #      4  +-----------+
            raise PlanFormatError('The plan contains rooms that are not fully enclosed by walls.')
        elif len(distinct_mapped_rooms) == 0:
            # The room section was enclosed by walls the previous line
            # We create a new room in chairs
            room = self._create_new_room()
        elif len(distinct_mapped_rooms) == 1:
            # The section is mapped to a room existing in the previous line
            room = distinct_mapped_rooms[0]
        else:
            # The section joins several rooms. This happens if the geometry of the flat is as follows:
            #
            #                 1    +-----------------------+
            #                 2    |           |           |
            #                 3    |           +           |
            #                 4    |                       |
            #                 5    +-----------------------+
            # and we are in line 4.
            room = self._join_rooms(distinct_mapped_rooms)
        self.new_status[section_idx] = [room for k in self.new_status[section_idx]]
        return room

    def _create_new_room(self) -> int:
        """
        Appends a new rom to Dataframe chairs with empty/zero values and returns its index

        :return: index of new row
        """

        room_idx = self.chairs.index.max() + 1
        # default values: None for name and zero chairs of each type
        self.chairs.loc[room_idx, :] = [None, 0, 0, 0, 0]
        return room_idx

    def _join_rooms(self, rooms: Set[int]):
        """
        Joins the given rooms in Dataframe chair, adds the discovered chair and asserts they have a common name
        :param rooms:
        :return:
        """
        joint_room = rooms[0]
        for room in rooms[1:]:

            # update chairs
            self.chairs.loc[joint_room, CHAIR_TYPES] += self.chairs.loc[room, CHAIR_TYPES]

            # update room name if necessary
            room_name = self.chairs.loc[room, 'room']
            if room_name is not None:
                if self.chairs.loc[joint_room, 'room'] is None:
                    self.chairs.loc[joint_room, 'room'] = room_name
                else:
                    msg = 'Multiple names were given for the same room:' \
                          + ' {} and {}!'.format(room_name, self.chairs.loc[joint_room, 'room'])
                    raise PlanFormatError(msg)
        return joint_room

    def _update_room_name_and_chairs(self, room_section: Tuple[int, int], line: str, room: int) -> None:
        """
        Extracts potential room name and chair data from room section and updates self.chairs accordingly.
        :param room_section: tuple containing start and endpoint of room section in line
        :param line: currently processed line of input file
        :param room: mapped room of line segment
        :return: None
        """
        # end points are walls and must not be considered
        segment_idx = slice(room_section[0] + 1, room_section[1])
        segment_string = line[segment_idx]

        # update room name
        room_name = re.search('([a-z ]+)', segment_string)
        if room_name:
            if self.chairs.loc[room, 'room'] is None:
                room_name = room_name.string
                self.chairs.loc[room, 'room'] = room_name
                segment_string.replace(room_name, '')
            else:
                msg = 'Multiple Names for the same room: {} and {}!'.format(room_name, self.chairs.loc[room, 'room'])
                raise PlanFormatError(msg)

        # update chairs
        self.chairs.loc[room, CHAIR_TYPES] += [segment_string.count(chair) for chair in CHAIR_TYPES]

    def _perform_final_validation(self) -> None:
        """
        Performs a final validation of the read data to assert that the process finished successfully.
        The following points assessed
            1. the status line must contain only None and wall values, otherwise a room was not closed
            2. detected rooms without names are removed from self.chairs. They could for instance constitute
                a yard etc. (Any are enclosed by walls that does not belong ot the flat.) If such an area contains
                chairs an error is raised.

        :return: None

        :raises PlanFormatError:
        """

        # if wrongly detected rooms (yards etc.) contain chairs rise an error
        # otherwise remove them from self.chairs and self.status
        no_rooms_idx = self.chairs['name'] == None
        if not (self.chairs.loc[no_rooms_idx, CHAIR_TYPES] == 0).all().all():
            raise PlanFormatError('The plan contains chairs outside any room.')
        else:
            self.chairs = self.chairs.drop(columns=no_rooms_idx)
            self.status = [None if k in no_rooms_idx else k for k in self.status]

        # validate that all rooms were closed, i.e. last row contains None elements or -1 (i.e. walls)
        if not all([k in [None, -1] for k in self.status]):
            raise PlanFormatError('The plan contains rooms that are not fully enclosed by walls.')

        # validate that room names are unique
        if any(self.chairs.duplicated(subset='name')):
            raise PlanFormatError('The plan contains different rooms with the same name.')
